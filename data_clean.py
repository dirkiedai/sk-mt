import argparse
import sys
import os 


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing Data for All-in-One Model Training")

    parser.add_argument("--input", type=str, help="input data")
    parser.add_argument("--output", type=str, help="output data")
    parser.add_argument("--subset", type=str, help="subset data")  
    parser.add_argument("--max-t", type=int, help="the maximum number of tm")
    parser.add_argument("--task", type=str, help="task")

    return parser.parse_args()


def main(args):

    if args.task == "translation":

        tm_src = ["" for _ in range(args.max_t)]
        tm_tgt = ["" for _ in range(args.max_t)]

        with open(args.input, 'r') as fr:

            for line in fr.readlines():
                items = line.strip('\n').split('\t')
                src, tgt = items[0], items[1]

                tm_count = 0
                for index in range(args.max_t):
                    src_tm_index = index * 2 + 2
                    tgt_tm_index = index * 2 + 3
                    if src_tm_index < len(items):
                        src_tm_len = len(items[src_tm_index])
                        tgt_tm_len = len(items[tgt_tm_index])
                        if tgt_tm_len > 0.5 * src_tm_len and tgt_tm_len < 2 * src_tm_len and src_tm_len != 1 and tgt_tm_len != 1:
                            tm_src[tm_count] += items[src_tm_index] + "\n" 
                            tm_tgt[tm_count] += items[tgt_tm_index] + "\n"
                            tm_count += 1
                    else:
                        tm_src[tm_count] += " \n"
                        tm_tgt[tm_count] += " \n"
                        tm_count += 1

                if tm_count < args.max_t:
                    for _ in range(args.max_t - tm_count):
                        tm_src[tm_count + _] += " \n"
                        tm_tgt[tm_count + _] += " \n"

            for i in range(args.max_t):
                src_path = os.path.join(args.output, f'{args.subset}{i + 1}.de')
                with open(src_path, 'w') as fw:
                    fw.write(tm_src[i])
                tgt_path = os.path.join(args.output, f'{args.subset}{i + 1}.en')
                with open(tgt_path, 'w') as fw:
                    fw.write(tm_tgt[i])

    elif args.task == 'language_modeling':
        tm_src = ["" for _ in range(args.max_t)]

        with open(args.input, 'r') as fr:

            for line in fr.readlines():
                items = line.strip('\n').split('\t')

                tm_count = 0
                for index in range(args.max_t):
                    src_tm_index = index + 1

                    if src_tm_index < len(items):
                        src_tm_len = len(items[src_tm_index])
                        if src_tm_len != 1 :
                            tm_src[tm_count] += items[src_tm_index] + "\n" 
                            tm_count += 1
                    else:
                        tm_src[tm_count] += " \n"
                        tm_count += 1

                if tm_count < args.max_t:
                    for _ in range(args.max_t - tm_count):
                        tm_src[tm_count + _] += " \n"

            for i in range(args.max_t):
                src_path = os.path.join(args.output, f'{args.subset}{i + 1}.tokens')
                with open(src_path, 'w') as fw:
                    fw.write(tm_src[i])


if __name__ == '__main__':
    main(parse_args())
