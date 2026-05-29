from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


_LR_PER_PHASE = {1: 1e-3, 2: 1e-4, 3: 1e-5}


def _dataset_hash(npz_path: str | Path) -> str:
    with open(npz_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def _make_mlflow_callback(epoch_offset: int = 0):
    """Fabrique un callback Keras qui logue dans MLflow à chaque epoch."""
    import tensorflow as tf

    class MLflowEpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None) -> None:
            import mlflow
            if logs:
                mlflow.log_metrics(
                    {k: float(v) for k, v in logs.items()},
                    step=epoch_offset + epoch,
                )

    return MLflowEpochLogger()


class Trainer:
    """Gère les 3 phases d'entraînement avec TensorBoard + MLflow en parallèle.

    Usage
    -----
    trainer = Trainer(
        model, enc_ear,
        experiment   = 'temp007_bs32',
        dataset_path = 'dataset/preprocessed_dataset.npz',
        base_dir     = 'checkpoints/',
    )
    trainer.run_phase(1, ds.train, ds.val, epochs=50)
    trainer.run_phase(2, ds.train, ds.val, epochs=30)
    trainer.end()
    trainer.plot_history()

    TensorBoard
    -----------
    tensorboard --logdir checkpoints/exp_.../logs --port 6007

    MLflow UI
    ---------
    mlflow ui --backend-store-uri checkpoints/mlruns --port 5000
    → http://localhost:5000
    """

    def __init__(
        self,
        model,
        ear_encoder,
        experiment:   str             = 'default',
        dataset_path: str | Path | None = None,
        base_dir:     str | Path      = 'checkpoints/',
    ) -> None:
        self._model      = model
        self._enc_ear    = ear_encoder
        self._experiment = experiment
        self._ds_hash    = _dataset_hash(dataset_path) if dataset_path else 'nohash'

        exp_name       = f'exp_{experiment}_{self._ds_hash}'
        self._save_dir = Path(base_dir) / exp_name
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._history: dict[str, list] = {'loss': [], 'val_loss': [], 'phase': []}

        # ------------------------------------------------------------------
        # MLflow
        # ------------------------------------------------------------------
        self._mlflow_active = False
        try:
            import mlflow

            mlflow.set_tracking_uri(f"sqlite:///{Path(base_dir) / 'mlruns.db'}")
            mlflow.set_experiment(experiment)
            self._run = mlflow.start_run(run_name=exp_name)
            mlflow.log_params({
                'experiment'  : experiment,
                'dataset_hash': self._ds_hash,
                'temperature' : float(model.temperature),
            })
            self._mlflow_active = True
            print(f'  MLflow run ID   : {self._run.info.run_id}')

        except ImportError:
            print('  MLflow non installé — versioning désactivé.')
            print('  Pour activer : pip install mlflow\n')

        print(f'  Expérience      : {exp_name}')
        print(f'  Répertoire      : {self._save_dir}')

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def run_phase(
        self,
        phase:    int,
        train_ds,
        val_ds,
        epochs:   int,
    ) -> None:
        import tensorflow as tf

        if phase not in _LR_PER_PHASE:
            raise ValueError(f'phase doit être 1, 2 ou 3 — reçu {phase}')

        print(f'\n{"="*60}')
        print(f'  PHASE {phase}  —  lr={_LR_PER_PHASE[phase]}  —  {epochs} epochs max')
        print(f'{"="*60}\n')

        if self._mlflow_active:
            import mlflow
            mlflow.log_params({
                f'phase{phase}_lr'    : _LR_PER_PHASE[phase],
                f'phase{phase}_epochs': epochs,
            })

        self._enc_ear.set_phase(phase)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(_LR_PER_PHASE[phase])
        )

        h = self._model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = epochs,
            callbacks       = self._make_callbacks(phase),
            verbose         = 0,
        )

        val_key = next(
            (k for k in h.history if 'loss' in k and k != 'loss'), None
        )

        n_epochs_ran = len(h.history['loss'])
        self._history['loss']     += h.history['loss']
        self._history['val_loss'] += h.history[val_key] if val_key else []
        self._history['phase']    += [phase] * n_epochs_ran

        best_val = min(h.history[val_key]) if val_key else float('nan')
        print(f'\n  Phase {phase} terminée — {n_epochs_ran} epochs')
        print(f'  Meilleure val_loss : {best_val:.4f}')

        self._save_config(phase, n_epochs_ran, best_val)

    # ------------------------------------------------------------------
    # Sauvegarde JSON + artefacts MLflow
    # ------------------------------------------------------------------

    def _save_config(self, phase: int, epochs_ran: int, best_val: float) -> None:
        config = {
            'experiment'   : self._experiment,
            'dataset_hash' : self._ds_hash,
            'phase'        : phase,
            'epochs_ran'   : epochs_ran,
            'best_val_loss': round(best_val, 6),
            'lr'           : _LR_PER_PHASE[phase],
            'temperature'  : float(self._model.temperature),
            'timestamp'    : datetime.now().isoformat(timespec='seconds'),
        }

        config_path  = self._save_dir / f'phase{phase}_config.json'
        weights_path = self._save_dir / f'phase{phase}_best.weights.h5'

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        if self._mlflow_active:
            import mlflow
            mlflow.log_metric(f'phase{phase}_best_val_loss', best_val)
            mlflow.log_artifact(str(config_path))
            if weights_path.exists():
                mlflow.log_artifact(str(weights_path))

        print(f'  Config sauvegardée → {config_path}')

    # ------------------------------------------------------------------
    # Fermeture du run MLflow
    # ------------------------------------------------------------------

    def end(self) -> None:
        """Ferme le run MLflow. À appeler après le dernier run_phase()."""
        if self._mlflow_active:
            import mlflow
            mlflow.end_run()
            print(f'  Run MLflow fermé — ID : {self._run.info.run_id}')

    # ------------------------------------------------------------------
    # Callbacks  (TensorBoard + MLflow + EarlyStopping + ...)
    # ------------------------------------------------------------------

    def _make_callbacks(self, phase: int) -> list:
        import tensorflow as tf
        from tqdm.keras import TqdmCallback

        checkpoint_path = str(self._save_dir / f'phase{phase}_best.weights.h5')
        log_dir         = str(self._save_dir / 'logs' / f'phase{phase}')
        epoch_offset    = len(self._history['loss'])

        callbacks = [TqdmCallback(verbose=1)]

        # MLflow — logue métriques epoch par epoch
        if self._mlflow_active:
            callbacks.append(_make_mlflow_callback(epoch_offset=epoch_offset))

        # TensorBoard — visualisation live + histogrammes
        try:
            import tensorboard  # noqa: F401
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir        = log_dir,
                histogram_freq = 1,
                update_freq    = 'epoch',
            ))
        except ImportError:
            print('  TensorBoard non installé — logs visuels désactivés.')

        callbacks += [
            tf.keras.callbacks.EarlyStopping(
                monitor              = 'val_loss',
                patience             = 10,
                restore_best_weights = True,
                verbose              = 1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor  = 'val_loss',
                factor   = 0.5,
                patience = 5,
                min_lr   = 1e-7,
                verbose  = 1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath          = checkpoint_path,
                monitor           = 'val_loss',
                save_best_only    = True,
                save_weights_only = True,
                verbose           = 0,
            ),
        ]

        return callbacks

    # ------------------------------------------------------------------
    # Visualisation locale
    # ------------------------------------------------------------------

    def plot_history(self) -> None:
        import matplotlib.pyplot as plt

        if not self._history['loss']:
            print("Aucun historique — lance run_phase() d'abord.")
            return

        loss     = self._history['loss']
        val_loss = self._history['val_loss']
        phases   = self._history['phase']
        epochs   = range(1, len(loss) + 1)

        _, ax = plt.subplots(figsize=(12, 5))
        ax.plot(epochs, loss,     label='train loss', color='steelblue')
        ax.plot(epochs, val_loss, label='val loss',   color='tomato')

        prev = phases[0]
        for i, p in enumerate(phases):
            if p != prev:
                ax.axvline(x=i + 1, linestyle='--', alpha=0.6,
                           label=f'début phase {p}')
                prev = p

        ax.set_xlabel('Epoch')
        ax.set_ylabel('NT-Xent Loss')
        ax.set_title(f"Historique — exp_{self._experiment}_{self._ds_hash}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()