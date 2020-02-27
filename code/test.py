import numpy as np
import tensorflow as tf

from Model import Model
from data import Poem

poem_generator = Poem()
dictionary, reversed_dictionary = poem_generator.data_set()

def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return reversed_dictionary[sample]

def Test():
    # 定义输入的只有一个字词，然后根据上一个字词推测下一个词的位置
    input_data = tf.placeholder(tf.int32, [1, None])
    # 输入和输出的尺寸为1
    input_size = output_size = len(reversed_dictionary) + 1
    # 定义模型
    model = Model(X=input_data, batch_size=1, input_size=input_size, output_size=output_size)
    # 获取模型的输出参数
    _, last_state, probs, initial_state = model.results()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        print("generate...")
       # saver.restore(session, './model/poetry.module-140')
        # 起始字符是'['，
        x = np.array([list(map(dictionary.get, '['))])
        # 运行初始0状态
        state_ = session.run(initial_state)
        word = poem = '['
        # 结束字符是']'
        while word != ']':
            # 使用上一级的state和output作为输入
            probs_, state_ = session.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
            poem += word
            # 获取词语的id
            x = np.zeros((1, 1))
            x[0, 0] = dictionary[word]
        print(poem)

def TestHead(characters):
    input_data = tf.placeholder(tf.int32, [1, None])
    input_size = output_size = len(reversed_dictionary) + 1
    model = Model(X=input_data, batch_size=1, input_size=input_size, output_size=output_size)
    _, last_state, probs, initial_state = model.results()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        print("generate...")

        flag = 1
        endSign = {-1: "，", 1: "。"}
        poem = ''
        x = np.array([list(map(dictionary.get, '['))])
        state_ = session.run(initial_state)

        for word in characters:
            if reversed_dictionary[word] == None:
                print("不认识这个字")
                exit(0)
            flag = -flag
            while word not in [']', '，', '。', ' ', '？', '！']:
                poem += word
                x = np.array([list(map(dictionary.get, word))])
                probs_, state_ = session.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
                word = to_word(probs_)
            poem += endSign[flag]
            if endSign[flag] == '。':
                probs_, state_ = session.run([probs, last_state], feed_dict={input_data: np.array([list(map(dictionary.get, '。'))]), initial_state: state_})
                poem += '\n'
            else:
                probs_, state_ = session.run([probs, last_state], feed_dict={input_data: np.array([list(map(dictionary.get, '，'))]), initial_state: state_})                
        print(characters)
        print(poem)