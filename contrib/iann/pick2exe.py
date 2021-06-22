from qpt.executor import CreateExecutableModule

if __name__ == '__main__':
    module = CreateExecutableModule(work_dir="./iann",
                                    launcher_py_path="./iann/exe.py",
                                    save_path="./out")
    module.make()
