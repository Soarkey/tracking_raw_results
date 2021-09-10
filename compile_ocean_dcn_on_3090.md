# 3090编译Ocean可变形卷积 / Compile Ocean DCN on 3090

## 1.环境 / Environment

- Ubuntu 20
- Python 3.7
- CUDA 11.4
- PyTorch 1.9.0+cu111

## 2.修改 / Modification

找到`lib/models/dcn/src/deform_conv.py`文件, 将所有的`AT_CHECK`替换为`AT_ASSERT`, 同时将所有的`.view`操作替换为`.contiguous().view`
操作, 之后重新执行编译`python setup.py develop`.

Found file `lib/models/dcn/src/deform_conv.py`, replace all `AT_CHECK` and `.view` operations with `AT_ASSERT` and `.contiguous().view`, then recompile `python setup.py develop`.

## 3.可能碰到的问题及解决 / References

### > ValueError: Unknown CUDA arch (8.6) or GPU not supported

- 解决 / Solution

  将conda环境所在文件夹中的`cpp_extension.py`内容从:

    ```python
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
    ])
    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5']
    ```

  改为

    ```python
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere', '8.0;8.6+PTX'),
    ])
    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0', '8.6']
    ```

  区别在于：增加了8.6的支持, 3090就是属于sm86架构.

- see solution: https://blog.csdn.net/ng323/article/details/116940299)

### > undefined symbol: THPVariableClass

- 原因： 在导入某些和pytorch有关的第三方包时，如果先导入第三方包，容易发生这种错误，正确的做法是首先导入pytorch。
- see solution: https://blog.csdn.net/slow122/article/details/116030717