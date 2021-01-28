#Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import paddle.fluid as fluid


def nccl2_prepare(args, startup_prog, main_prog):
    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    t = fluid.DistributeTranspiler(config=config)

    envs = args.dist_env

    t.transpile(
        envs["trainer_id"],
        trainers=','.join(envs["trainer_endpoints"]),
        current_endpoint=envs["current_endpoint"],
        startup_program=startup_prog,
        program=main_prog)


def pserver_prepare(args, train_prog, startup_prog):
    config = fluid.DistributeTranspilerConfig()
    config.slice_var_up = args.split_var
    t = fluid.DistributeTranspiler(config=config)
    envs = args.dist_env
    training_role = envs["training_role"]

    t.transpile(
        envs["trainer_id"],
        program=train_prog,
        pservers=envs["pserver_endpoints"],
        trainers=envs["num_trainers"],
        sync_mode=not args.async_mode,
        startup_program=startup_prog)
    if training_role == "PSERVER":
        pserver_program = t.get_pserver_program(envs["current_endpoint"])
        pserver_startup_program = t.get_startup_program(
            envs["current_endpoint"],
            pserver_program,
            startup_program=startup_prog)
        return pserver_program, pserver_startup_program
    elif training_role == "TRAINER":
        train_program = t.get_trainer_program()
        return train_program, startup_prog
    else:
        raise ValueError(
            'PADDLE_TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
        )


def nccl2_prepare_paddle(trainer_id, startup_prog, main_prog):
    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(
        trainer_id,
        trainers=os.environ.get('PADDLE_TRAINER_ENDPOINTS'),
        current_endpoint=os.environ.get('PADDLE_CURRENT_ENDPOINT'),
        startup_program=startup_prog,
        program=main_prog)


def prepare_for_multi_process(exe, build_strategy, train_prog):
    # prepare for multi-process
    trainer_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    if num_trainers < 2: return

    build_strategy.num_trainers = num_trainers
    build_strategy.trainer_id = trainer_id
    # NOTE(zcd): use multi processes to train the model,
    # and each process use one GPU card.
    startup_prog = fluid.Program()
    nccl2_prepare_paddle(trainer_id, startup_prog, train_prog)
    # the startup_prog are run two times, but it doesn't matter.
    exe.run(startup_prog)
