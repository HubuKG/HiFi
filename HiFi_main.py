import torch
import mmkgc
from mmkgc.module.model import HiFi
from mmkgc.adv.GSAG import MultiGenerator
from mmkgc.data import TrainDataLoader, TestDataLoader
from mmkgc.config import Tester, AdvMixTrainer
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling
from args import get_args

if __name__ == "__main__":
    # Get the arguments passed during the execution and print them.
    args = get_args()
    print(args)

    # Set random seeds for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data loader for training.
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )

    # Data loader for testing.
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")

    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')

    # Define the model.
    kge_score = HiFi(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        epsilon=2.0,
        margin=args.margin,
        text_emb=text_emb,
        img_emb=img_emb
    )
    print(kge_score)

    # Define the loss function.
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
    )

    # Define the adversarial generator.
    adv_generator = MultiGenerator(
        noise_dim=64,
        structure_dim=2 * args.dim,
        img_dim=2 * args.dim,
    ).cuda()

    # Train the model.
    trainer = AdvMixTrainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam',
        generator=adv_generator,
        lrg=args.lrg,
        mu=args.mu
    )
    trainer.run()

    # Save the trained model.
    kge_score.save_checkpoint(args.save)

    # Test the model.
    kge_score.load_checkpoint(args.save)
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
