from fastNLP import Vocabulary
import torch
ATTR_NULL_TAG = 'null'


# TODO: 目前只支持BMOES
def input_with_span_attr(datasets, vocabs):
    datasets['train'].apply_field(lambda x: list(map(lambda y: y[0], x)), field_name='target',
                                  new_field_name='span_label')
    if 'dev' in datasets:
        datasets['dev'].apply_field(lambda x: list(map(lambda y: y[0], x)), field_name='target',
                                    new_field_name='span_label')
    datasets['test'].apply_field(lambda x: list(map(lambda y: y[0], x)), field_name='target',
                                 new_field_name='span_label')

    datasets['train'].apply_field(lambda x: list(map(lambda y: y[2:] if y[0] in ['S', 'B'] else ATTR_NULL_TAG, x)),
                                  field_name='target', new_field_name='attr_start_label')
    if 'dev' in datasets:
        datasets['dev'].apply_field(lambda x: list(map(lambda y: y[2:] if y[0] in ['S', 'B'] else ATTR_NULL_TAG, x)),
                                    field_name='target', new_field_name='attr_start_label')
    datasets['test'].apply_field(lambda x: list(map(lambda y: y[2:] if y[0] in ['S', 'B'] else ATTR_NULL_TAG, x)),
                                 field_name='target', new_field_name='attr_start_label')

    datasets['train'].apply_field(lambda x: list(map(lambda y: y[2:] if y[0] in ['S', 'E'] else ATTR_NULL_TAG, x)),
                                  field_name='target', new_field_name='attr_end_label')
    if 'dev' in datasets:
        datasets['dev'].apply_field(lambda x: list(map(lambda y: y[2:] if y[0] in ['S', 'E'] else ATTR_NULL_TAG, x)),
                                    field_name='target', new_field_name='attr_end_label')
    datasets['test'].apply_field(lambda x: list(map(lambda y: y[2:] if y[0] in ['S', 'E'] else ATTR_NULL_TAG, x)),
                                 field_name='target', new_field_name='attr_end_label')

    span_label_vocab = Vocabulary()
    attr_label_vocab = Vocabulary()
    span_label_vocab.from_dataset(datasets['train'], field_name='span_label')
    attr_label_vocab.from_dataset(datasets['train'], field_name='attr_start_label')
    vocabs['span_label'] = span_label_vocab
    vocabs['attr_label'] = attr_label_vocab
    print(f"span label: {span_label_vocab.word2idx.keys()}")
    print(f"attr label: {attr_label_vocab.word2idx.keys()}")
    return datasets, vocabs


# extract entities by start and end positions
def extract_kvpairs_by_start_end(start_seq, end_seq, neg_tag):
    assert len(start_seq) == len(end_seq)
    pairs = []
    i = 0
    while i < len(start_seq):
        s_tag = start_seq[i]
        if s_tag not in neg_tag:
            for j, e_tag in enumerate(end_seq[i:]):
                # if j > 0 and start_seq[i+j] != neg_tag or j + 1 > 30:
                if j > 0 and start_seq[i+j] not in neg_tag:
                    i = i + j - 1
                    break
                if s_tag == e_tag:
                    pairs.append(((i, i + j + 1), s_tag))
                    i = i + j
                    break
        i += 1
    return pairs


def convert_attr_seq_to_ner_seq(attr_start_tensor, attr_end_tensor, vocabs, tagscheme='BMOES'):
    batch_size = attr_start_tensor.size(0)
    seq_len = attr_start_tensor.size(1)
    device = attr_start_tensor.device
    label_vocab = vocabs['label']
    attr_label_vocab = vocabs['attr_label']
    attr_start_pred_tag = attr_start_tensor.detach().cpu().numpy()
    attr_end_pred_tag = attr_end_tensor.detach().cpu().numpy()
    pred_label = []
    pred_label_text = []
    attr_neg_tags = [attr_label_vocab.to_index(w) for w in ['<pad>', '<unk>', ATTR_NULL_TAG]]
    for idx in range(batch_size):
        attr_start_seq, attr_end_seq = attr_start_pred_tag[idx], attr_end_pred_tag[idx]
        pairs = extract_kvpairs_by_start_end(attr_start_seq, attr_end_seq, attr_neg_tags)
        ner_seqs = ['O'] * seq_len
        for pair in pairs:
            (spos, epos), attr = pair
            attr_name = attr_label_vocab.to_word(attr)
            if tagscheme == 'BMOES':
                if epos - spos == 1:
                    ner_seqs[spos] = 'S-' + attr_name
                else:
                    ner_seqs[spos] = 'B-' + attr_name
                    ner_seqs[epos - 1] = 'E-' + attr_name
                    ner_seqs[spos + 1: epos - 1] = ['M-' + attr_name] * (epos - spos - 2)
            elif tagscheme == 'BIO':
                if epos - spos == 1:
                    ner_seqs[spos] = 'B-' + attr_name
                else:
                    ner_seqs[spos] = 'B-' + attr_name
                    ner_seqs[spos + 1: epos - 1] = ['I-' + attr_name] * (epos - spos - 2)
            else:
                raise ValueError('Unknown tagscheme: {}!'.format(tagscheme))
        try:
            unknown_idx = label_vocab.to_index('O')  # 因为_tag是根据attr生成，可能不在label_alphabet里
            label_keys = list(label_vocab.word2idx.keys())
            pred_label.append(
                [label_vocab.to_index(_tag) if _tag in label_keys else unknown_idx for _tag in ner_seqs])
        except:
            # print("Error in {}".format(ner_seqs))
            print('Error')
        pred_label_text.append(ner_seqs)
    pred_variable = torch.tensor(pred_label, requires_grad=False).long().to(device)
    return pred_variable
