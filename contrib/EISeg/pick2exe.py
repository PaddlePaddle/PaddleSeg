from qpt.executor import CreateExecutableModule

if __name__ == "__main__":
    module = CreateExecutableModule(
        work_dir="./EISeg", launcher_py_path="./EISeg/exe.py", save_path="./out"
    )
    module.make()
