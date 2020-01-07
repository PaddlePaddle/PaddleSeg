import os
from paddle import fluid

def load_fp16_vars(executor, dirname, program):
    load_dirname = os.path.normpath(dirname)

    def _if_exist(var):
        name = var.name[:-7] if var.name.endswith('.master') else var.name
        b = os.path.exists(os.path.join(load_dirname, name))
        if not b and isinstance(var, fluid.framework.Parameter):
            print("===== {} not found ====".format(var.name))
        return b

    load_prog = fluid.Program()
    load_block = load_prog.global_block()
    vars = list(filter(_if_exist, program.list_vars()))

    for var in vars:
        new_var = fluid.io._clone_var_in_block_(load_block, var)
        name = var.name[:-7] if var.name.endswith('.master') else var.name
        file_path = os.path.join(load_dirname, name)
        load_block.append_op(
            type='load',
            inputs={},
            outputs={'Out': [new_var]},
            attrs={
                'file_path': file_path,
                'load_as_fp16': var.dtype == fluid.core.VarDesc.VarType.FP16
            })

    executor.run(load_prog)