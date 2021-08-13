from qpt.executor import CreateExecutableModule as CEM

module = CEM(work_dir="eiseg",
             launcher_py_path="eiseg/exe.py",
             save_path="out")

# 开始打包
module.make()