from qpt.executor import CreateExecutableModule

if __name__ == "__main__":
    module = CreateExecutableModule(
        work_dir="contrib/EISeg", 
        launcher_py_path="contrib/EISeg/eiseg/exe.py", 
        save_path="contrib/out"
    )
    module.make()