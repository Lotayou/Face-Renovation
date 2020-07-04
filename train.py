import sys,torch
from collections import OrderedDict
# change config file for ablation study
from options.config_hifacegan import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
torch.backends.cudnn.benchmark = True

opt = TrainOptions()

if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    if epoch%opt.upsample_phase_epoch_fq==opt.upsample_phase_epoch_fq-1:
        opt.train_phase += 1
    # 20200211: Will more training phase be helpful?
    opt.train_phase = min(opt.train_phase, opt.max_train_phase)
    trainer.update_train_phase(train_phase=opt.train_phase)

    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            sys_im = trainer.get_latest_generated()
            h, w = sys_im.size()[-2:]
            in_im = torch.nn.functional.interpolate(data_i['label'], size=(h, w))
            visuals = OrderedDict([('input_label', in_im),
                                   ('synthesized_image', sys_im),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
