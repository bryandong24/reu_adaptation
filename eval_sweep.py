"""
Systematic evaluation of Push-F checkpoints across different time limits.
Saves all results to a JSON file for reporting.
"""
import os, json, torch, dill, hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace

CHECKPOINTS = {
    "epoch_250": "data/outputs/2026.02.23/23.35.33_train_diffusion_unet_image_pushf_image/checkpoints/epoch=0250-test_mean_score=0.880.ckpt",
    "epoch_400": "data/outputs/2026.02.23/23.35.33_train_diffusion_unet_image_pushf_image/checkpoints/epoch=0400-test_mean_score=0.874.ckpt",
    "epoch_500": "data/outputs/2026.02.23/23.35.33_train_diffusion_unet_image_pushf_image/checkpoints/latest.ckpt",
}

TIME_LIMITS = {
    "30s": 300,
    "45s": 450,
    "60s": 600,
    "90s": 900,
}

results = {}

for ckpt_name, ckpt_path in CHECKPOINTS.items():
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)

    for time_name, max_steps in TIME_LIMITS.items():
        print(f"\n--- {ckpt_name} @ {time_name} (max_steps={max_steps}) ---")

        output_dir = f"data/pushf_eval_sweep/{ckpt_name}_{time_name}"
        os.makedirs(output_dir, exist_ok=True)

        cfg.task.env_runner.max_steps = max_steps

        workspace = cls(cfg, output_dir=output_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.ema_model
        device = torch.device('cuda:0')
        policy.to(device)
        policy.eval()

        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
        runner_log = env_runner.run(policy)

        # Extract results
        test_rewards = {}
        for k, v in runner_log.items():
            if k.startswith("test/sim_max_reward"):
                test_rewards[k] = v

        perfect = sum(1 for v in test_rewards.values() if v == 1.0)
        above_80 = sum(1 for v in test_rewards.values() if v >= 0.8)
        above_50 = sum(1 for v in test_rewards.values() if v >= 0.5)
        test_mean = runner_log.get("test/mean_score", 0)
        train_mean = runner_log.get("train/mean_score", 0)

        key = f"{ckpt_name}_{time_name}"
        results[key] = {
            "checkpoint": ckpt_name,
            "time_limit": time_name,
            "max_steps": max_steps,
            "test_mean_score": test_mean,
            "train_mean_score": train_mean,
            "perfect_seeds": perfect,
            "above_80_seeds": above_80,
            "above_50_seeds": above_50,
            "total_seeds": len(test_rewards),
            "per_seed_rewards": test_rewards,
        }

        print(f"  test_mean_score: {test_mean:.4f}")
        print(f"  Perfect: {perfect}/{len(test_rewards)}, >0.8: {above_80}/{len(test_rewards)}, >0.5: {above_50}/{len(test_rewards)}")

# Save all results
out_path = "data/pushf_eval_sweep/all_results.json"
json.dump(results, open(out_path, 'w'), indent=2, sort_keys=True)
print(f"\nAll results saved to {out_path}")

# Print summary table
print(f"\n{'='*70}")
print(f"SUMMARY TABLE")
print(f"{'='*70}")
print(f"{'Checkpoint':<12} | {'Time':<5} | {'Mean Score':<11} | {'Perfect':<9} | {'>0.8':<6} | {'>0.5':<6}")
print("-" * 70)
for key in sorted(results.keys()):
    r = results[key]
    print(f"{r['checkpoint']:<12} | {r['time_limit']:<5} | {r['test_mean_score']:<11.4f} | {r['perfect_seeds']:<4}/{r['total_seeds']:<4} | {r['above_80_seeds']:<4}  | {r['above_50_seeds']:<4}")
