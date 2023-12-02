from transformers.trainer import *


# The substitue funtion for trainer._save_checkpoint
def _save_checkpoint(self, model, trial, metrics=None):
    # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    # want to save except FullyShardedDDP.
    # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    # Save model checkpoint
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)
    self.save_model(output_dir, _internal_call=True)
    if self.is_deepspeed_enabled:
        # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
        # config `stage3_gather_16bit_weights_on_model_save` is True

        ##### LLMBox deletion
        # self.model_wrapped.save_checkpoint(output_dir)
        pass

    # Save optimizer and scheduler
    if self.fsdp or self.is_fsdp_enabled:
        if self.is_fsdp_enabled:
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        else:
            # FSDP has a different interface for saving optimizer states.
            # Needs to be called on all ranks to gather all states.
            # full_optim_state_dict will be deprecated after Pytorch 2.2!
            full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)
            torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))

    if is_torch_tpu_available():
        xm.rendezvous("saving_optimizer_states")
        xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        with warnings.catch_warnings(record=True) as caught_warnings:
            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
    elif is_sagemaker_mp_enabled():
        opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
        smp.barrier()
        if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
            smp.save(
                opt_state_dict,
                os.path.join(output_dir, OPTIMIZER_NAME),
                partial=True,
                v3=smp.state.cfg.shard_optimizer_state,
            )
    elif self.args.should_save and not self.is_deepspeed_enabled and not (self.fsdp or self.is_fsdp_enabled):
        # deepspeed.save_checkpoint above saves model/optim/sched
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

    # Save SCHEDULER & SCALER
    is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
        self.lr_scheduler, DeepSpeedSchedulerWrapper
    )
    if (
        self.args.should_save and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
        and not is_torch_tpu_available()
    ):
        with warnings.catch_warnings(record=True) as caught_warnings:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        reissue_pt_warnings(caught_warnings)

    # Determine the new best metric / best model checkpoint
    if metrics is not None and self.args.metric_for_best_model is not None:
        metric_to_check = self.args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics[metric_to_check]

        operator = np.greater if self.args.greater_is_better else np.less
        if (
            self.state.best_metric is None or self.state.best_model_checkpoint is None
            or operator(metric_value, self.state.best_metric)
        ):
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir

    # Save the Trainer state
    if self.args.should_save:
        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    # Save RNG state in non-distributed training
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
            rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
        else:
            rng_states["cuda"] = torch.cuda.random.get_rng_state()

    if is_torch_tpu_available():
        rng_states["xla"] = xm.get_rng_state()

    if is_torch_npu_available():
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            rng_states["npu"] = torch.npu.random.get_rng_state_all()
        else:
            rng_states["npu"] = torch.npu.random.get_rng_state()

    # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
    # not yet exist.
    os.makedirs(output_dir, exist_ok=True)

    ##### LLMBox deletion
    # if self.args.world_size <= 1:
    #     torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
    # else:
    #     torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

    if self.args.push_to_hub:
        self._push_from_checkpoint(output_dir)

    # Maybe delete some older checkpoints.
    if self.args.should_save:
        self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
