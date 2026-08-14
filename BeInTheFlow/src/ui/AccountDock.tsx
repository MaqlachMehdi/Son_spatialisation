import { useState } from "react";
import { useAuthStore } from "../store/authStore";

function UserIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

export default function AccountDock() {
  const [collapsed, setCollapsed] = useState(true);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const { user, status, error, login, register, logout, clearError } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        await register(email, password, displayName);
      }
      setPassword("");
    } catch {
      // l'erreur est déjà posée dans le store, rien de plus à faire ici
    } finally {
      setSubmitting(false);
    }
  };

  const switchMode = (next: "login" | "register") => {
    setMode(next);
    clearError();
  };

  return (
    <div className="account-dock">
      <button
        className="sidebar-toggle settings-toggle"
        onClick={() => setCollapsed((c) => !c)}
        aria-label={collapsed ? "Ouvrir le compte" : "Replier le compte"}
        aria-expanded={!collapsed}
      >
        <UserIcon />
      </button>

      <div className={`settings-panel ${collapsed ? "collapsed" : ""}`}>
        <div className="settings-panel-inner">
          <div className="sidebar-header">
            <h1>Compte</h1>
          </div>

          {status === "authenticated" && user ? (
            <>
              <p className="field-group-label">
                Connecté en tant que <strong>{user.displayName || user.email}</strong>
              </p>
              <button className="add-btn" onClick={() => logout()}>
                Se déconnecter
              </button>
            </>
          ) : (
            <>
              <div className="account-tabs">
                <button
                  className={`account-tab ${mode === "login" ? "active" : ""}`}
                  onClick={() => switchMode("login")}
                  type="button"
                >
                  Se connecter
                </button>
                <button
                  className={`account-tab ${mode === "register" ? "active" : ""}`}
                  onClick={() => switchMode("register")}
                  type="button"
                >
                  Créer un compte
                </button>
              </div>

              <form onSubmit={handleSubmit} className="account-form">
                {mode === "register" && (
                  <label className="field">
                    <span>Nom affiché</span>
                    <input
                      type="text"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                    />
                  </label>
                )}
                <label className="field">
                  <span>Email</span>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Mot de passe</span>
                  <input
                    type="password"
                    required
                    minLength={8}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                </label>
                {error && <p className="error">{error}</p>}
                <button className="play-btn" type="submit" disabled={submitting}>
                  {submitting ? "…" : mode === "login" ? "Se connecter" : "Créer le compte"}
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}