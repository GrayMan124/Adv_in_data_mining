import numpy as np


def jaccard_similarity(u1, u2):
  u1 = np.where(u1 > 0)[0]
  u2 = np.where(u2 > 0)[0]

  u1 = set(u1.tolist())
  u2 = set(u2.tolist())

  return len(u1.intersection(u2)) / len(u1.union(u2))


def filling_vectors(u1, u2, movies_num):
  if len(u1) < movies_num: u1 = np.concatenate((u1, np.zeros(shape=(movies_num - len(u1),))), axis=0).astype(int)
  if len(u2) < movies_num: u2 = np.concatenate((u2, np.zeros(shape=(movies_num - len(u2),))), axis=0).astype(int)
  return u1, u2

def cosine_similarity(u1, u2):
  return np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2))


def discrete_cosine_similarity(u1, u2):
  u1[np.where(u1 > 0)[0]] = 1
  u2[np.where(u2 > 0)[0]] = 1
  return cosine_similarity(u1, u2)


movies_num = 6
u1 = np.array([0, 5, 4, 0, 3])
u2 = np.array([5, 4, 3])
u1, u2 = filling_vectors(u1, u2, movies_num)

print(f'Jaccard Similarity: {jaccard_similarity(u1, u2)}')

print(f'Cosine Similarity: {cosine_similarity(u1, u2)}')

print(f'Discrete Cosine Similarity: {discrete_cosine_similarity(u1, u2)}')