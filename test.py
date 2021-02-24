from paths import *
clip_len = 230
msra_limit_len = 250
sentence_num = 0

def create_cliped_file(fp):
    global sentence_num
    f = open(fp,'r',encoding='utf-8')
    fp_out = fp + '_clip2'
    f_out = open(fp_out,'w',encoding='utf-8')
    now_example_len = 0

    lines = f.readlines()
    last_line_split = None
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 0:
            last_line_split = None
            now_example_len = 0
            print('', file=f_out)
            sentence_num += 1
            continue
        if line_split[1][0].lower() == 'b' and last_line_split is not None and now_example_len>clip_len:
            sentence_num += 1
            print('',file=f_out)
        print(line,end='',file=f_out) # line包含'\n'
        now_example_len += 1
        if line_split[0] in ['。','！','？'] and line_split[1] == 'O' and now_example_len>clip_len:
            print('',file=f_out)
            now_example_len = 0
            sentence_num += 1
        elif ((line_split[0] in ['，','；'] or (now_example_len>1 and last_line_split[0] == '…' and line_split[0] == '…'))
                 and line_split[1] == 'O' and now_example_len>clip_len):
            print('',file=f_out)
            now_example_len = 0
            sentence_num += 1
        elif line_split[1][0].lower() == 'o' and now_example_len>clip_len:
            print('',file=f_out)
            now_example_len = 0
            sentence_num += 1

        last_line_split = line_split

    f_out.close()
    f_check = open(fp_out,'r',encoding='utf-8')
    lines = f_check.readlines()
    cliped_examples = [[]]
    now_example = cliped_examples[0]
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 0:
            cliped_examples.append([])
            now_example = cliped_examples[-1]
        else:
            now_example.append(line.strip())

    check = 0
    max_length = 0
    for example in cliped_examples:
        if len(example)>msra_limit_len:
            # print(len(example), example)
            print(len(example),''.join(map(lambda x:x.split('\t')[0],example)))
            check = 1

        max_length = max(max_length,len(example))

    print('最长的句子有:{}'.format(max_length))

    if check == 0:
        print('没句子超过{}的长度'.format(msra_limit_len))

create_cliped_file('{}/msra_train_bio.txt'.format(msra_ner_cn_path))
print(sentence_num)
sentence_num = 0
# create_cliped_file('{}/msra_test_bio.txt'.format(msra_ner_cn_path))
print(sentence_num)