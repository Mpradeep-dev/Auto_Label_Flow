import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
import { SectionLabel } from "@/components/layout/SectionLabel";
import type { IntegrationStatus } from "@/types";

// Kaggle and Roboflow are account-wide connections, not scoped to a
// project (PLAN follow-on "Credential scope: Global") — this panel lives
// on the top-level /settings route precisely so it's reachable without
// picking a project first. See pages/IntegrationsPage.tsx.

function FieldError({ error }: { error: unknown }) {
  if (!error) return null;
  const message = error instanceof ApiError ? error.message : (error as Error).message;
  return <p className="mt-2 text-xs text-accent">{message}</p>;
}

function StatusPill({ status }: { status: IntegrationStatus }) {
  return (
    <span
      className={`px-2 py-1 text-[10px] font-bold uppercase tracking-widest ${
        status.connected ? "bg-ink text-paper" : "bg-muted text-ink/60"
      }`}
    >
      {status.connected ? "Connected" : "Not connected"}
    </span>
  );
}

function KaggleCard({ status }: { status: IntegrationStatus }) {
  const queryClient = useQueryClient();
  const [username, setUsername] = useState("");
  const [key, setKey] = useState("");

  const connectMutation = useMutation({
    mutationFn: () => api.connectKaggle({ username: username.trim(), key: key.trim() }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
      setUsername("");
      setKey("");
    },
  });
  const disconnectMutation = useMutation({
    mutationFn: () => api.disconnectKaggle(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["integrations"] }),
  });

  return (
    <div className="border-2 border-ink p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-lg font-bold uppercase tracking-tight">Kaggle</p>
        <StatusPill status={status} />
      </div>
      <p className="mb-4 text-sm text-ink/60">
        Powers the KAGGLE training provider — push a dataset version and train a base model on Kaggle's free GPU
        quota instead of local hardware. Local training always stays available regardless of this.
      </p>

      {status.connected ? (
        <div className="flex items-center justify-between border-t border-ink/20 pt-4">
          <p className="tabular text-xs uppercase tracking-widest text-ink/50">
            {status.identifier}
            {status.verified_at && ` · verified ${new Date(status.verified_at).toLocaleString()}`}
          </p>
          <button
            onClick={() => disconnectMutation.mutate()}
            disabled={disconnectMutation.isPending}
            className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-orange hover:text-paper hover:border-orange disabled:opacity-40"
          >
            Disconnect
          </button>
        </div>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (username.trim() && key.trim()) connectMutation.mutate();
          }}
          className="space-y-3 border-t border-ink/20 pt-4"
        >
          <div className="flex flex-wrap gap-3">
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="KAGGLE USERNAME"
              className="min-w-[200px] flex-1 border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <input
              value={key}
              onChange={(e) => setKey(e.target.value)}
              type="password"
              placeholder="API KEY"
              className="min-w-[200px] flex-1 border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <button
              type="submit"
              disabled={!username.trim() || !key.trim() || connectMutation.isPending}
              className="border-2 border-ink bg-ink px-6 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
            >
              {connectMutation.isPending ? "Verifying…" : "Connect"}
            </button>
          </div>
          <FieldError error={connectMutation.error} />
          {status.last_error && !connectMutation.isError && (
            <p className="text-xs text-accent">Last attempt failed: {status.last_error}</p>
          )}
          <p className="text-xs text-ink/40">
            From your Kaggle account → Settings → API → Create New Token. The key is stored on this server only.
          </p>
        </form>
      )}
    </div>
  );
}

function ModalCard({ status }: { status: IntegrationStatus }) {
  const queryClient = useQueryClient();
  const [tokenId, setTokenId] = useState("");
  const [tokenSecret, setTokenSecret] = useState("");

  const connectMutation = useMutation({
    mutationFn: () => api.connectModal({ token_id: tokenId.trim(), token_secret: tokenSecret.trim() }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
      setTokenId("");
      setTokenSecret("");
    },
  });
  const disconnectMutation = useMutation({
    mutationFn: () => api.disconnectModal(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["integrations"] }),
  });

  return (
    <div className="border-2 border-ink p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-lg font-bold uppercase tracking-tight">Modal</p>
        <StatusPill status={status} />
      </div>
      <p className="mb-4 text-sm text-ink/60">
        Powers the MODAL training provider — train on Modal's serverless GPU cloud with per-second billing.
        $30/month free credits included. Local training always stays available regardless of this.
      </p>

      {status.connected ? (
        <div className="flex items-center justify-between border-t border-ink/20 pt-4">
          <p className="tabular text-xs uppercase tracking-widest text-ink/50">
            Token: {status.identifier}
            {status.verified_at && ` · verified ${new Date(status.verified_at).toLocaleString()}`}
          </p>
          <button
            onClick={() => disconnectMutation.mutate()}
            disabled={disconnectMutation.isPending}
            className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-orange hover:text-paper hover:border-orange disabled:opacity-40"
          >
            Disconnect
          </button>
        </div>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (tokenId.trim() && tokenSecret.trim()) connectMutation.mutate();
          }}
          className="space-y-3 border-t border-ink/20 pt-4"
        >
          <div className="flex flex-wrap gap-3">
            <input
              value={tokenId}
              onChange={(e) => setTokenId(e.target.value)}
              placeholder="TOKEN ID"
              className="min-w-[200px] flex-1 border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <input
              value={tokenSecret}
              onChange={(e) => setTokenSecret(e.target.value)}
              type="password"
              placeholder="TOKEN SECRET"
              className="min-w-[200px] flex-1 border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <button
              type="submit"
              disabled={!tokenId.trim() || !tokenSecret.trim() || connectMutation.isPending}
              className="border-2 border-ink bg-ink px-6 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
            >
              {connectMutation.isPending ? "Verifying…" : "Connect"}
            </button>
          </div>
          <FieldError error={connectMutation.error} />
          {status.last_error && !connectMutation.isError && (
            <p className="text-xs text-accent">Last attempt failed: {status.last_error}</p>
          )}
          <p className="text-xs text-ink/40">
            From modal.com/settings → API Tokens → Create new token. The credentials are stored on this server only.
          </p>
        </form>
      )}
    </div>
  );
}

function RoboflowCard({ status }: { status: IntegrationStatus }) {
  const queryClient = useQueryClient();
  const [apiKey, setApiKey] = useState("");
  const [defaultWorkspace, setDefaultWorkspace] = useState("");

  const connectMutation = useMutation({
    mutationFn: () =>
      api.connectRoboflow({ api_key: apiKey.trim(), default_workspace: defaultWorkspace.trim() || undefined }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
      setApiKey("");
      setDefaultWorkspace("");
    },
  });
  const disconnectMutation = useMutation({
    mutationFn: () => api.disconnectRoboflow(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["integrations"] }),
  });

  return (
    <div className="border-2 border-ink p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-lg font-bold uppercase tracking-tight">Roboflow</p>
        <StatusPill status={status} />
      </div>
      <p className="mb-4 text-sm text-ink/60">
        Both directions: pull an existing Roboflow project version in as a new dataset here, or push a cut dataset
        version out to a Roboflow project. Both actions live on the Dataset and Export pages once connected.
      </p>

      {status.connected ? (
        <div className="flex items-center justify-between border-t border-ink/20 pt-4">
          <p className="tabular text-xs uppercase tracking-widest text-ink/50">
            Default workspace: {status.identifier}
            {status.verified_at && ` · verified ${new Date(status.verified_at).toLocaleString()}`}
          </p>
          <button
            onClick={() => disconnectMutation.mutate()}
            disabled={disconnectMutation.isPending}
            className="border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest hover:bg-orange hover:text-paper hover:border-orange disabled:opacity-40"
          >
            Disconnect
          </button>
        </div>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (apiKey.trim()) connectMutation.mutate();
          }}
          className="space-y-3 border-t border-ink/20 pt-4"
        >
          <div className="flex flex-wrap gap-3">
            <input
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              type="password"
              placeholder="API KEY"
              className="min-w-[200px] flex-1 border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <input
              value={defaultWorkspace}
              onChange={(e) => setDefaultWorkspace(e.target.value)}
              placeholder="DEFAULT WORKSPACE (optional)"
              className="min-w-[200px] flex-1 border-2 border-ink bg-paper px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <button
              type="submit"
              disabled={!apiKey.trim() || connectMutation.isPending}
              className="border-2 border-ink bg-ink px-6 py-2 text-xs font-bold uppercase tracking-widest text-paper hover:bg-orange disabled:opacity-40"
            >
              {connectMutation.isPending ? "Verifying…" : "Connect"}
            </button>
          </div>
          <FieldError error={connectMutation.error} />
          {status.last_error && !connectMutation.isError && (
            <p className="text-xs text-accent">Last attempt failed: {status.last_error}</p>
          )}
          <p className="text-xs text-ink/40">
            From app.roboflow.com → Settings → Roboflow API. Leave workspace blank to use your account's default.
          </p>
        </form>
      )}
    </div>
  );
}

export function IntegrationsSection({ sectionIndex = 2 }: { sectionIndex?: number }) {
  const integrationsQuery = useQuery({ queryKey: ["integrations"], queryFn: () => api.listIntegrations() });
  const byProvider = Object.fromEntries((integrationsQuery.data ?? []).map((s) => [s.provider, s]));
  const kaggle = byProvider["KAGGLE"] ?? { provider: "KAGGLE", connected: false, identifier: null, verified_at: null, last_error: null };
  const modal = byProvider["MODAL"] ?? {
    provider: "MODAL",
    connected: false,
    identifier: null,
    verified_at: null,
    last_error: null,
  };
  const roboflow = byProvider["ROBOFLOW"] ?? {
    provider: "ROBOFLOW",
    connected: false,
    identifier: null,
    verified_at: null,
    last_error: null,
  };

  return (
    <section>
      <SectionLabel index={sectionIndex}>Integrations</SectionLabel>
      <div className="grid max-w-5xl grid-cols-1 gap-6 lg:grid-cols-2">
        <KaggleCard status={kaggle} />
        <ModalCard status={modal} />
        <RoboflowCard status={roboflow} />
      </div>
    </section>
  );
}
