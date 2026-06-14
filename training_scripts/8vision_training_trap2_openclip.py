import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description="TRAP2 for OpenCLIP (ConvNeXt).")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to train on.")
    parser.add_argument('--device', type=int, default=None, help="GPU id (e.g., 0), or -1 for CPU. Default: auto.")
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lambda_reg', type=float, default=0.01)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--warm_up', type=int, default=500)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--config', dest='config_name', type=str, default='8vision_train_openclip_convnext')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='peft_merging')
    parser.add_argument('--rand_alpha_min', type=float, default=0.05)
    parser.add_argument('--rand_alpha_max', type=float, default=2.0)
    parser.add_argument('--rand_alpha_weight', type=str, default="inv",
                        choices=["inv", "inv_sqrt", "none"])
    args = parser.parse_args()

    training_config = vars(args)
    training_config['peft_type'] = 'lora'
    training_config['num_workers'] = 8
    training_config['default_batch_size'] = 32
    training_config['use_target_val'] = 1

    base = importlib.import_module("training_scripts.8vision_training_trap2")
    base.CONFIG_NAME = args.config_name
    loss = base.train_functional(training_config, device_override=args.device)
    print(loss)


if __name__ == "__main__":
    main()
