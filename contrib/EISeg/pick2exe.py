from qpt.executor import CreateExecutableModule as CEM

module = CEM(work_dir=r"E:\dataFiles\github\PaddleSeg\contrib\EISeg\eiseg",
             launcher_py_path=r"E:\dataFiles\github\PaddleSeg\contrib\EISeg\eiseg\exe.py",
             save_path=r"E:\dataFiles\github\PaddleSeg\contrib\EISeg\out")

# 开始打包
module.make()