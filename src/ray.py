def initialize_ray():
    import ray
    address = f"127.0.0.1:10002"
    runtime_env = {
        "working_dir": "./",
        "excludes": ["training_save_dir", "helm_charts", "tests", ".git", "run", "run/artifacts", "run/wandb", "artifacts", "wandb", "*.mp4", "*.ckpt", "*.png", "*.npy"],
    }
    ray.client(address).env(runtime_env).connect()
