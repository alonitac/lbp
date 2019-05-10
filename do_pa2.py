###############################################################################
# HUJI 67800 PMAI 2018/19 - Programming Assignment 2
# Original authors: Ya Le, Billy Jun, Xiaocheng Li, Yiftach Beer
###############################################################################
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *


def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices
    return values:
        G: generator matrix
        H: parity check matrix
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H


def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde
        yTilde_i is obtained by flipping y_i with probability epsilon
    '''
    yTilde = y ^ np.random.choice([0, 1], p=[1-epsilon, epsilon], size=y.shape)
    return yTilde


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructFactorGraph(yTilde, H, epsilon):
    '''
    :param - yTilde: observed codeword
        type: numpy.ndarray containing 0's and 1's
        shape: 2N
    :param - H parity check matrix
             type: numpy.ndarray
             shape: N x 2N
    :param epsilon - the probability that each bit is flipped to its complement
    return G FactorGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors
    '''

    N, M = H.shape
    graph = FactorGraph(numVar=M, numFactor=N+M)
    graph.var = range(M)

    # Add unary factors
    for i in graph.var:
        val = np.array([epsilon, 1-epsilon])
        if not yTilde[i]:
            val = np.flip(val, 0)
        factor = Factor(scope=[i], card=[2], val=val, name='unary-{}'.format(i))
        graph.addFactor(factor)

    # Add parity factors
    for i in range(N):
        scope = list(np.where(H[i, :] == 1)[0])
        card = [2] * len(scope)
        val = np.empty(tuple(card))
        assignments = itertools.product([0, 1], repeat=len(scope))

        for a in assignments:
            val[a] = (sum(a) + 1) % 2
        factor = Factor(scope=scope, card=[2]*len(scope), val=val, name='parity-{}'.format(i))
        graph.addFactor(factor)
    return graph


def do_part_b():
    yTilde = np.array([[1, 1, 1, 1, 1, 1]]).reshape(6,1)
    H = np.array([[0, 1, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0],
                  [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    graph = constructFactorGraph(yTilde, H, epsilon)
    ytest1 = np.array([[0, 1, 1, 0, 1, 0]]).reshape(6,1)  # invalid
    ytest2 = np.array([[0, 1, 0, 1, 1, 0]]).reshape(6,1)  # invalid
    ytest3 = np.array([[1, 1, 1, 1, 0, 0]]).reshape(6,1)  # valid
    print(
        graph.evaluateWeight(ytest1),
        graph.evaluateWeight(ytest2),
        graph.evaluateWeight(ytest3))


def do_part_c():
    '''
    In part c, we provide you an all-zero initialization of message x, you should
    apply noise on y to get yTilde, and then do loopy BP to obtain the
    marginal probabilities of the unobserved y_i's.
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    error = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    yTilde = applyChannelNoise(y, error)
    graph = constructFactorGraph(yTilde, H, error)
    graph.runParallelLoopyBP(10)
    marginals = np.empty((2*N, 2))
    for var in range(2*N):
        marginals[var, :] = graph.estimateMarginalProbability(var)

    plt.figure()
    plt.title('Part C: Marginals for all-zeros input, error={}'.format(error))
    plt.plot(range(len(y)), marginals, '.-')
    plt.savefig('part_c.png', bbox_inches='tight')


def do_part_de(numTrials, error, iterations=50):
    '''
    param - numTrials: how many trials we repeat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)

    plt.figure()
    plt.title('Part D/E: Hamming distances, error={}'.format(error))
    for trial in range(numTrials):
        print('Trial number', trial)
        ##############################################################
        # Todo: your code starts here
        # apply noise, construct the graph
        # run loopy while retrieving the marginal MAP after each iteration
        # calculate Hamming distances and plot
        #
        # ....
        #
        # plt.plot(hamming_distances)
    plt.grid(True)
    plt.savefig('part_de_{}.png'.format(error), bbox_inches='tight')
    ##############################################################


def do_part_fg(error):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = np.load('image.npy')

    N = G.shape[1]
    x = img.reshape(N, 1)
    y = encodeMessage(x, G)
    yTilde = applyChannelNoise(y, error)

    plt.figure()
    plt.title('Part F/G: Image reconstruction, error={}'.format(error))
    show_image(yTilde, 0, 'Input')

    ##############################################################
    # Todo: your code starts here
    #
    # ....
    #
    ##############################################################
    # plot_iters = np.array([0, 1, 3, 5, 10, 20, 30])
    # for i, result in enumerate(results[plot_iters]):
    #     show_image(result, i+1, 'Iter {}'.format(plot_iters[i]))
    plt.savefig('part_fg_{}.png'.format(error), bbox_inches='tight')
    ################################################################


def show_image(output, loc, title, num_locs=8):
    image = output.flatten()[:len(output)//2]
    image_radius = int(np.sqrt(image.shape))
    image = image.reshape((image_radius, image_radius))
    ax = plt.subplot(1, num_locs, loc + 1)
    ax.set_title(title)
    ax.imshow(image)
    ax.axis('off')


if __name__ == "__main__":
    # print('Doing part (b): Should see 0.0, 0.0, >0.0')
    # do_part_b()

    print('Doing part (c):')
    do_part_c()

    # print('Doing part (d):')
    # do_part_de(10, 0.06)
    #
    # print('Doing part (e):')
    # do_part_de(10, 0.08)
    # do_part_de(10, 0.10)
    #
    # print('Doing part (f):')
    # do_part_fg(0.06)
    #
    # print('Doing part (g):')
    # do_part_fg(0.10)
    #
    # print('All done.')
    # plt.show()
