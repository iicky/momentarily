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
  // Weekly cron (Sun 05:00 UTC) starts the trainer container. R2 credentials
  // are forwarded from Worker secrets into the container's environment, where
  // training/r2_client.py reads them ahead of the (absent) murk vault.
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
