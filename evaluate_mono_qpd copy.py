if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='Interp', help="name your experiment")
    parser.add_argument('--tsubame', action='store_true', help="when you run on tsubame")
    parser.add_argument('--restore_ckpt', type=str, default=None, help="restore checkpoint")
    args = parser.parse_args()

    if args.tsubame:
        dcl = get_train_config(args.exp_name)
        conf = dcl.tsubame()
    else:
        conf = get_train_config(args.exp_name)
    
    if args.restore_ckpt:
        conf.restore_ckpt_mono_qpd = args.restore_ckpt
    print(conf)

    train(conf)
