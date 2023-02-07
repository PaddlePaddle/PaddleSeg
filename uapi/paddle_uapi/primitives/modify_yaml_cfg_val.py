# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import io
import re
import contextlib

import yaml

from .primitive import Primitive


class ModifyYAMLCfgVal(Primitive):
    _DELIMITER = '.'
    _CFG_VAL_PATTERN = r'(?<=\: ).*$'

    def __init__(self,
                 cfg_desc,
                 val,
                 file_path=None,
                 comm_key=None,
                 match_strategy='in_dict',
                 yaml_loader=None):
        super().__init__(comm_key)
        self.file_path = file_path
        self.cfg_desc = cfg_desc
        self.keys = self._parse_cfg_desc(self.cfg_desc)
        self.val = val
        if match_strategy not in ('by_text', 'in_dict'):
            raise ValueError("Unsupported matching strategy")
        self.match_strategy = match_strategy
        if yaml_loader is not None:
            self.yaml_loader = yaml_loader
        else:
            self.yaml_loader = yaml.Loader

    def do(self, comm=None):
        with self._create_context(comm=comm):
            pass

    def set(self, comm=None):
        ctx = self._create_context(comm=comm)
        return ctx

    @contextlib.contextmanager
    def _create_context(self, comm):
        if self.file_path is None:
            file_path = comm[self.comm_key]
        else:
            file_path = self.file_path

        try:
            if self.match_strategy == 'in_dict':
                self._match_in_dict(file_path, comm)
            elif self.match_strategy == 'by_text':
                self._match_by_text(file_path, comm)
        except Exception as e:
            raise RuntimeError(
                f"Unable to match '{self.cfg_desc}' in {file_path} .\nThe specific error information is: {str(e)}"
            )

        yield comm

    def _match_by_text(self, file_path, comm):
        # HACK:FIXME: Current implementation rewrites the whole file,
        # which is inefficient for large files.
        # NOTE: We assume that there are no duplicate keys in the YAML file.
        prev_indents = None
        all_matched = False
        with open(file_path, 'r') as f, io.StringIO() as out:
            # Match keys sequentially
            num_keys = len(self.keys)
            for i, key in enumerate(self.keys, 1):
                try:
                    line = next(f)
                except StopIteration:
                    break
                if self._is_tag_literally(line):
                    # NOTE: In 'by_text' mode we ignore all tags
                    pass
                else:
                    # XXX: We count all starting blanks here
                    num_indents = len(line) - len(line.lstrip())
                    if prev_indents is not None:
                        # We are matching sub-keys 
                        # According to YAML rule, a sub-key should have more indents
                        if num_indents <= prev_indents:
                            break
                    matched = self._match_key_literally(key, line)
                    if matched:
                        prev_indents = num_indents
                        if i == num_keys:
                            # All keys matched
                            line = self._update_line(line)
                            all_matched = True
                out.write(line)
            if not all_matched:
                # TODO: More friendly error logs
                raise RuntimeError
            else:
                # Write the remaining lines
                for line in f:
                    out.write(line)
                # Overwrite the file
                with open(file_path, 'w') as f:
                    f.write(out.getvalue())

    def _parse_cfg_desc(self, cfg_desc):
        # TODO: Support more complex configs (currently only key-val pairs)
        return cfg_desc.split(self._DELIMITER)

    def _is_tag_literally(self, line):
        return line.lstrip().startswith('!')

    def _match_key_literally(self, key, line):
        return line.lstrip().startswith(key)

    def _update_line(self, line):
        return re.sub(self._CFG_VAL_PATTERN, str(self.val), line)

    def _match_in_dict(self, file_path, comm):
        whole_dict = self._load_yaml(file_path)
        if not isinstance(whole_dict, dict):
            raise TypeError
        assert len(self.keys) >= 1
        dict_ = whole_dict
        for key in self.keys[:-1]:
            dict_ = dict_[key]
        # Update value
        dict_[self.keys[-1]] = self.val
        # Overwrite the file
        with open(file_path, 'w') as f:
            f.write(yaml.dump(whole_dict))

    def _load_yaml(self, yaml_file_path):
        with open(yaml_file_path, 'r') as f:
            cfg = yaml.load(f, Loader=self.yaml_loader)
        return cfg
