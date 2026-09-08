import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  TRAINER: DurableObjectNamespace<TrainerContainer>;
  R2_ACCOUNT_ID: string;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  R2_BUCKET: string;
}

export class TrainerContainer extends Container {
  // Batch job — no listening port. EM runs to completion and the container
  // exits; give the instance a generous idle window before it's reaped.
  override sleepAfter = "20m";

  override onStop(params: { exitCode: number; reason: string }): void {
    const level = params.exitCode === 0 ? "log" : "error";
    console[level](
      `trainer container stopped: exit=${params.exitCode} reason=${params.reason}`,
    );
  }
}

export default {
  // Weekly cron (Sun 05:00 UTC) that would start the trainer container —
  // CURRENTLY PAUSED: wrangler.toml sets `crons = []`, so Cloudflare never fires
  // this handler and the trainer is run by hand (`murk exec -- uv run python -m
  // training.train_em`). The handler stays wired so restoring `crons` is the
  // only change needed to resume. R2 credentials are forwarded from Worker
  // secrets into the container's environment, where training/r2_client.py reads
  // them ahead of the (absent) murk vault.
  //
  // RESUME CRITERION (derived from the 2026-09-04 shadow-HMM review of the live
  // model v1788229972; journal 2026-09-04 entry): the pause exists so this one
  // version accrues a clean eval window, and that review returned NO-GO because
  // the recovery arm was too thin and single-route to grade. Resume — un-pause
  // `crons` or run the manual train_em above — once the post-v1788229972 shadow
  // window has accrued at least 20 graded recovery incidents (MIN_RECOVERY_REGIMES,
  // training/eval.py — the low_sample floor that review's current-segment arm
  // missed at 19<20) spanning at least 3 distinct routes (that window's recovery
  // population was ~98% route H, 115/117 current-segment ticks, which collapsed
  // causal_skill=-1.70 to a single-route read, not a network statement). The
  // 20-incident floor is the review's own data-sufficiency threshold; the 3-route
  // span is an added condition against that single-route artifact (the review sets
  // no route floor of its own). Together they let the next review render a real
  // GO/NO-GO on recovery rather than "directional only". Un-pausing `crons` also
  // activates
  // .github/workflows/trainer-staleness-check.yml, which then alarms if a weekly
  // run fails to advance state/params.json trained_at.
  async scheduled(_event: ScheduledController, env: Env): Promise<void> {
    const container = getContainer(env.TRAINER, "weekly");
    await container.start({
      entrypoint: ["python", "-m", "training.train_em"],
      envVars: {
        R2_ACCOUNT_ID: env.R2_ACCOUNT_ID,
        R2_ACCESS_KEY_ID: env.R2_ACCESS_KEY_ID,
        R2_SECRET_ACCESS_KEY: env.R2_SECRET_ACCESS_KEY,
        R2_BUCKET: env.R2_BUCKET,
      },
      enableInternet: true,
    });
  },
} satisfies ExportedHandler<Env>;
