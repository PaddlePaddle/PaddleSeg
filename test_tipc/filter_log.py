#!/usr/bin/env python

import re
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_log_path", type=str)
    parser.add_argument("--out_log_path", type=str)
    parser.add_argument("--skip_iters", type=int, default=0)
    parser.add_argument("--pattern", type=str, default=r'epoch: (\d+)')
    args = parser.parse_args()

    pat = re.compile(args.pattern)
    prev_val = None
    with open(args.in_log_path, 'r') as fi, open(args.out_log_path, 'w') as fo:
        for line in fi:
            match = pat.search(line)
            if match is not None:
                groups = match.groups()
                if len(groups) != 1:
                    raise ValueError
                curr_val = groups[0]
                assert curr_val != None
                if curr_val != prev_val:
                    prev_val = curr_val
                    cnt = 0
                else:
                    cnt += 1
                if cnt >= args.skip_iters:
                    fo.write(line)
            else:
                fo.write(line)
                prev_val = None
                cnt = 0
