#!/usr/bin/env python
# -*- coding:UTF-8 -*-


####################################
# test about return a == b #
# def judgement(x1, x2):
#     return x1 == x2
#
#
# a = [1, 1]
# b = [1, 2]
# c = [1, 1]
#
# judgement(a, b)
#
# # print(a == b, (a == c).sum())

#####################################
# test about timer #
# import time
#
# class AA():
#
#     def __init__(self):
#         self.acc = 0
#         self.tic()
#
#     def tic(self):
#         self.t0 = time.time()
#
#     def toc(self):
#         diff = time.time()-self.t0
#         return diff
#
#     def hold(self):
#         self.acc += self.toc()
#
#
# A = AA()
# time.sleep(5)
# print(A.hold())
# time.sleep(1)
# print(A.acc)


####################################
# test about   return a >= b ####
def terminate(epoch):
    return epoch >= 20

print(terminate(1))