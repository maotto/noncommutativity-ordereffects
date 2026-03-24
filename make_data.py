from typing import Optional
from itertools import product
import math
import random
import numpy as np


def d1(
        order: list, #list of question order EX: [2,3,4,1,5] for 5 questions
        goodness_scores=None, #list of specific goodness scores from real data (rather than random)
        rng=None, #Seed for the random list of public probabilities 
):
    """ Takes a sequence of given or randomely generated likeability scores, and updates subsequent scores 
    dependent on the current score such that it is either raised or lowered by the average difference 
    between the two scores - thus making outcomes "more even". This rule, known as "even-handedness", typically 
    comes into play with questions containing similar objects; such as "Do you have a good opinion of Bill Clinton?" and "Do you
    have a good opinion of Al Gore?" The probability of asnwering "yes" to one of these questions depends on which one is asked first
    and the likeability scores of the individuals. This is described more in detail in Ref. https://www.jstor.org/stable/3078697. 

    :param order: list of question order EX: [2,3,4,1,5] for 5 questions
    :param goodness_scores: Input scores on the "goodness" of a person. If none, scores are random.
    :param rng: Numpy random number generator.
    :return: The probability distribution over bitstrings where "000" corresponds to "no, no, no".
    """ 
    n_questions = len(order)
    if goodness_scores == None:
        goodness_scores = []
        for i in range(0, n_questions):
            n = rng.random()
            goodness_scores.append(n)

    scorelist = [goodness_scores[i] for i in order]

    for count, (i, j) in enumerate(zip(scorelist[:-1], scorelist[1:])):
        mean_diff = (i - j) / 2
        if mean_diff < 0:
            scorelist[count + 1] = j + mean_diff
        if mean_diff > 0:
            scorelist[count + 1] = j + mean_diff
        else:
            pass

    bin_str = [''.join(p) for p in product('10', repeat=n_questions)]
    bin_str.sort(key=lambda s: s.count('1'))
    prob_dist = {}

    for string in bin_str:
        total_prob = 1
        for (s, score) in zip(string, scorelist):
            if s == "0":
                prob = 1 - score
            if s == "1":
                prob = score
            total_prob = total_prob * prob
        prob_dist[string] = total_prob

    return np.asarray(list(prob_dist.values()))

def d2(
    order_input: list, #list of question order EX: [2,3,4,1,5] for 5 questions
    rescale_coefficient: float, #portion amount to raise or lower the probabilitiy 
    seed = None, #Seed for the random list of public probabilities
    consistent_base_probabilities=False
) -> dict: 
    """ Takes a sequence of general to specific ranking questions with a corresponding random list of probabilities for public answers. If the order is changed
    such that a more specific question preceeds a more general question, the answer to the more general question will obtain a 45% increase in the probability that the 
    answer is yes. This dataset is based off of real data regarding school bullying, where we see 45% increases of students answering yes to being bullied
    if they are asked about a specific type of bullying first. This is found in the study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5965535/#:~:text=A%20randomized%20experiment%20(n%20%3D%205%2C951,several%20widely%20used%20bullying%20surveys.

        :param order_input: list of question orders starting at an index of zero. 
        :param seed: random seed if the goodness_scores are randomely generated. 
        :param rescale_coefficient: portion amount to raise or lower the probabilitiy
        :param consistent_base_probabilities: sort answers probabilities according to given order before applying order effect
        :return:  The probability distribution over bitstrings where "000" corresponds to "no, no, no".
    """
    n_questions = len(order_input)
    question_specifications = [i for i in np.arange(0, 1, 1 / n_questions)]  # General is Low, Specific is High
    # e.g. [0, 0.33, 0.66] for 3 questions. Signifies how concrete the question is.
    question_specifications = [question_specifications[i] for i in
                               order_input]  # sort concreteness values according to given order
    group_answers = []
    random.seed(seed)  # seed determines how subjects would answer question 1...N individually
    group_answers_unaltered_by_order_effect = []
    for i in range(0, n_questions):
        n = random.random() #0 is No, 1 is Yes
        group_answers.append(n)
        group_answers_unaltered_by_order_effect.append(n)
        # The seed is applied before the group_answers definition for-loop and the actual order is not taken into
        # account yet, so group_answers contains the default/"base" answer probabilities for question 0, 1 and 2.

    if consistent_base_probabilities:
        # sort the answering probability based on the order that the questions were asked
        group_answers = [group_answers[idx] for idx in order_input]

    for count, (i,j) in enumerate(zip(question_specifications[:-1], question_specifications[1:])):
        # for pairs of preceding and following question indices (i and j)
        # count is increasing [0,1]; i and j taken from question_specifications are the concreteness values of the
        # questions asked in the queried order
        if i > j:
            # if first question is more concrete:
            group_answers[count + 1] +=  (group_answers[count+1]) * rescale_coefficient  # answer "yes" is more likely;
            # in the bullying survey example this means:
            # primed with concrete bullying type, subjects are more likely to report to having been bullied

            # Ordered correctly? Here, we adjust the group_answers[count + 1] value.
            # With consistent_base_probabilities off:
            # These group_answers are not sorted according to the actual order in which they are asked.
            # The group_answers[0] value e.g. still reflects the value for question0 - no matter when q0 was actually
            # asked (with order_input=[1,2,0] it would be the last question).

            # The concreteness values stored in question_specifications however consider the actual order queried by the
            # method's caller (given as order_input).
            # Hence, when count==0, the i vs. j comparison is relevant for the question asked second (count+1), which is
            # the question number found in order_input[count+1].

            # With consistent_base_probabilities on:
            # Takes into account, which default answering probability is to be updated when questions 0,1,2 are asked in
            # the specified order, e.g. [question1, question2, question0].

        else: 
            pass  # case of the general to more specific order: keep answers generated above

    bin_str = [''.join(p) for p in product('10', repeat= n_questions)]  # 111 110 101 100 011 010 001 000
    bin_str.sort(key=lambda s: s.count('1'))  # 000 100 010 001 110 101 011 111 or ['00', '10', '01', '11']
    prob_dist = {}
    
    group_answers = group_answers / np.sum(group_answers)  # normalize to sum up to one: [0.0573  0.09546 0.84720]
    for string in bin_str:
        # one string is for example "010", meaning the user answered no, yes no
        total_prob = 1 
        for (s, score) in zip(string, group_answers):   # iterating over given answers and yes probability
            if s == "0":  # if answering pattern is "no" here at this evaluation time
                prob = 1 - score  # that probability is 1-score, e.g. 1-0.0573 in case "no" was indeed a common answer
            elif s == "1":
                prob = score
            else:
                raise ValueError(f"{s=}")
            total_prob = total_prob * prob  # joint probability
        prob_dist[string] = total_prob  # string is a pattern of answering, for example 0-0-1,
        # Question index view:
        # Is it intended to signify that subjects answer "no","no","yes" to *questions 0, 1 and 2*
        # (and they may have occurred in a different order during the survey)?
        #
        # Chronological view:
        # Or, that subjects have actually answered "no","no","yes" during the survey to the questions that were asked
        # in the given order?
        #
        # Depending on this view, an update of group_answers[order_input[count+1]] might be suitable in the
        # original code make_data.py#L86. Essentially, this would mean updating the correct index, but keeping
        # group_answers sorted by question index rather than chronologically.

    return np.asarray(list(prob_dist.values()))


def artificial_data_sampler2(order, rescale_coefficient=1.0, seed=123):
    """
    Wrapper for d2 to generate probability distributions.

    :param order: List[int], a permutation of question indices.
    :param rescale_coefficient: Float, scaling factor applied to probabilities.
    :param seed: Int, random seed for reproducibility.
    :return: Numpy array of probability distributions.
    """
    return d2(order, rescale_coefficient=rescale_coefficient, seed=seed)

