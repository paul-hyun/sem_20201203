# -*- coding:utf-8 -*-

import collections
import os
import re

#
# 말뭉치
#
corpus = """위키백과의 최상위 도메인이 .com이던 시절 ko.wikipedia.com에 구판 미디어위키가 깔렸으나 한글 처리에 문제가 있어 글을 올릴 수도 없는 이름뿐인 곳이었다. 2002년 10월에 새로운 위키 소프트웨어를 쓰면서 한글 처리 문제가 풀리기 시작했지만, 가장 많은 사람이 쓰는 인터넷 익스플로러에서는 인코딩 문제가 여전했다. 이런 이유로 초기에는 도움말을 옮기거나 쓰는 일에 어려움을 겪었다. 이런 어려움이 있었는데도 위키백과 통계로는, 2002년 10월에서 2003년 7월까지 열 달 사이에 글이 13개에서 159개로 늘었고 2003년 7월과 8월 사이에는 한 달 만에 159개에서 348개로 늘어났다. 2003년 9월부터는 인터넷 익스플로러의 인코딩 문제가 사라졌으며, 대한민국 언론에서도 몇 차례 위키백과를 소개하면서 참여자가 점증하리라고 예측했다. 참고로 한국어 위키백과의 최초 문서는 2002년 10월 12일에 등재된 지미 카터 문서이다.
2005년 6월 5일 양자장론 문서 등재를 기점으로 총 등재 문서 수가 1만 개를 돌파하였고 이어 동해 11월에 제1회 정보트러스트 어워드 인터넷 문화 일반 분야에 선정되었다. 2007년 8월 9일에는 한겨레21에서 한국어 위키백과와 위키백과 오프라인 첫 모임을 취재한 기사를 표지 이야기로 다루었다.
2008년 광우병 촛불 시위 때 생긴 신조어인 명박산성이 한국어 위키백과에 등재되고 이 문서의 존치 여부를 두고 갑론을박의 과정이 화제가 되고 각종 매체에도 보도가 되었다. 시위대의 난입과 충돌을 방지하기 위해 거리에 설치되었던 컨테이너 박스를 이명박 정부의 불통으로 풍자를 하기 위해 사용된 이 신조어는 중립성을 지켰는지와 백과사전에 올라올 만한 문서인지가 쟁점이 되었는데 일시적으로 사용된 신조어일 뿐이라는 주장과 이미 여러 매체에서 사용되어 지속성이 보장되었다는 주장 등 논쟁이 벌어졌고 다음 아고라 등지에서 이 항목을 존치하는 방안을 지지하는 의견을 남기기 위해 여러 사람이 새로 가입하는 등 혼란이 빚어졌다. 11월 4일에는 다음커뮤니케이션에서 글로벌 세계 대백과사전을 기증받았으며, 2009년 3월에는 서울특별시로부터 콘텐츠를 기증받았다. 2009년 6월 4일에는 액세스권 등재를 기점으로 10만 개 문서 수를 돌파했다.
2011년 4월 16일에는 대한민국에서의 위키미디어 프로젝트를 지원하는 모임을 결성할 것을 추진하는 논의가 이뤄졌고 이후 창립준비위원회 결성을 거쳐 2014년 10월 19일 창립총회를 개최하였으며, 최종적으로 2015년 11월 4일 사단법인 한국 위키미디어 협회가 결성되어 활동 중에 있다. 2019년 미국 위키미디어재단으로부터 한국 지역 지부(챕터)로 승인을 받았다.
2012년 5월 19일에는 보비 탬블링 등재를 기점으로 총 20만 개 문서가 등재되었고 2015년 1월 5일, Rojo -Tierra- 문서 등재를 기점으로 총 30만 개 문서가 등재되었다. 2017년 10월 21일에는 충청남도 동물위생시험소 문서 등재로 40만 개의 문서까지 등재되었다."""

#
# Char Tokenizer
#

# unique chars
chars = list(dict.fromkeys(list(corpus)))
print(len(chars), ':', chars)

# char to id
char_to_id = {'[PAD]': 0, '[UNK]': 1}  # PAD: token 수가 다를경우 길이를 맞춰 줌, UNK: vocab에 없는 token
for char in chars:
    char_to_id[char] = len(char_to_id)
print(len(char_to_id), ':', char_to_id)

# id to char
id_to_char = {i: char for char, i in char_to_id.items()}
print(len(char_to_id), ':', id_to_char)

# number of vocab
assert len(char_to_id) == len(id_to_char)
n_char_vocab = len(char_to_id)

# split by line
lines = corpus.split('\n')
print(lines)

# tokenize
tokens = [list(line) for line in lines]
print(tokens)

# token to id
char_ids = [[char_to_id[token] for token in line] for line in tokens]
print(char_ids)

# validate
id_chars = [[id_to_char[i] for i in line] for line in char_ids]
print(id_chars)

#
# Word Tokenizer
#

# unique words
words = list(dict.fromkeys(corpus.split()))
print(len(words), ':', words)

# word to id
word_to_id = {'[PAD]': 0, '[UNK]': 1}  # PAD: token 수가 다를경우 길이를 맞춰 줌, UNK: vocab에 없는 token
for word in words:
    word_to_id[word] = len(word_to_id)
print(len(word_to_id), ':', word_to_id)

# id to word
id_to_word = {i: word for word, i in word_to_id.items()}
print(len(word_to_id), ':', id_to_word)

# number of vocab
assert len(word_to_id) == len(id_to_word)
n_word_vocab = len(word_to_id)

# split by line
lines = corpus.split('\n')
print(lines)

# tokenize
tokens = [line.split() for line in lines]
print(tokens)

# token to id
word_ids = [[word_to_id[token] for token in line] for line in tokens]
print(word_ids)

# validate
id_words = [[id_to_word[i] for i in line] for line in word_ids]
print(id_words)


#
# BPE
#

def get_stats(vocab):
    """
    bi-gram 횟수를 구하는 함수
    :param vocab: vocab
    :return:
    """
    pairs = collections.defaultdict(int)  # 없는 단어는 0 리턴
    for word, freq in vocab.items():
        symbols = word.split()  # 값은 띄어쓰기로 구분 되어 있음
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq  # 이전 단어와 다음 단어의 빈도수
    return pairs


def merge_vocab(pair, v_in):
    """
    bi-gram을 합치는 함수
    :param pair: bi-gram pair
    :param v_in: 현재 vocab
    :return: bi-gram이 합쳐진 새로운 vocab
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)  # bigram을 substitute
        v_out[w_out] = v_in[word]
    return v_out


# 최초 말뭉치 빈도수
vocab = {'_ l o w': 5,
         '_ l o w e r': 2,
         '_ n e w e s t': 6,
         '_ w i d e s t': 3
         }

# 계산 횟수
num_merges = 10

# BPE 계산
for i in range(num_merges):
    print(f'######### {i} #########')
    pairs = get_stats(vocab)
    print(pairs)
    best = max(pairs, key=pairs.get)  # value 이 가장 큰 pair 조회
    print(best)
    vocab = merge_vocab(best, vocab)
    print(vocab)

#
# Sentence piece 학습
# https://github.com/google/sentencepiece
# https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
#

# 설치 (아래 명령을 이용해서 실행)
# pip install sentencepiece

# import 추가
import sentencepiece as spm

# data_dir 선언
data_dir = './data'
if not os.path.exists(data_dir):
    data_dir = '../data'
print(os.listdir(data_dir))


def train_sentencepiece(corpus, prefix, vocab_size=32000):
    """
    sentencepiece를 이용해서 vocab 생성
    :param corpus: 학습 corpus
    :param prefix: 저장할 이름 <prefix>.model, <prefix>.vocab 생성 됨
    :param vocab_size: 생성할 vocab 개수
    """
    spm.SentencePieceTrainer.train(
        f'--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}' +
        ' --model_type=unigram' +
        ' --max_sentence_length=999999' +  # 문장 최대 길이
        ' --pad_id=0 --pad_piece=[PAD]' +  # pad (0)
        ' --unk_id=1 --unk_piece=[UNK]' +  # unknown (1)
        ' --bos_id=2 --bos_piece=[BOS]' +  # begin of sequence (2)
        ' --eos_id=3 --eos_piece=[EOS]' +  # end of sequence (3)
        ' --user_defined_symbols=[SEP],[CLS],[MASK]')  # 기타 추가 토큰


# korean vocab 생성 (corpus (한국어 위키 데이터): data_dir/kowiki/kowiki.txt)
train_sentencepiece(os.path.join(data_dir, 'kowiki', 'kowiki.txt'), os.path.join(data_dir, 'ko_32000'), vocab_size=32000)

#
# Sentence piece 테스트
#

# vocab file print
with open(os.path.join(data_dir, 'ko_32000.vocab'), 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if 100 <= i:
            break

# load vocab
vocab_file = os.path.join(data_dir, 'ko_32000.model')
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

# vocab info major token
print(f'len: {len(vocab)}')
for i in range(100):
    print(f'{i:2d}: {vocab.id_to_piece(i)}')

# sentence to pieces (string -> tokens)
pieces = vocab.encode_as_pieces(corpus)
print(pieces[0:100])

# pieces to sentence (tokens -> string)
string = vocab.decode_pieces(pieces)
print(string)

# piece to id (tokens -> token_ids)
piece_ids = []
for piece in pieces:
    piece_ids.append(vocab.piece_to_id(piece))
print(piece_ids[0:100])

# sentence to ids (string -> token_ids)
ids = vocab.encode_as_ids(corpus)
print(ids[0:100])

# ids to sentence
string = vocab.decode_ids(ids)
print(string)

# id to piece
id_pieces = []
for i in ids:
    id_pieces.append(vocab.id_to_piece(i))
print(id_pieces[0:100])
