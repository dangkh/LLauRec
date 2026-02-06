import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from helper import getUser_Interaction
import random
from sklearn.manifold import TSNE
import numpy as np
import os
from matplotlib.patches import Circle

def getTop100SimilarUsers(user_id, user_embeddings, user_ids, topk = 100):
	# user_embeddings is numpy array of shape (num_users, embedding_dim)
	userembed = user_embeddings[user_id]
	similarities = []
	for other_user_id in user_ids:
		if other_user_id == user_id:
			continue
		other_user_embed = user_embeddings[other_user_id]
		# cosine_similarity = torch.nn.functional.cosine_similarity(userembed, other_user_embed, dim=0)
		cosine_similarity = np.dot(userembed, other_user_embed) / (np.linalg.norm(userembed) * np.linalg.norm(other_user_embed))
		similarities.append((other_user_id, cosine_similarity))

	similarities.sort(key=lambda x: x[1], reverse=True)
	top_k_similar_users = similarities[:topk]
	return top_k_similar_users

def countOverlap(user_interactions, user_id, similar_user_ids):
	user_items = set(user_interactions[user_id])
	overlap_counts = {}
	for sim_user_id in similar_user_ids:
		sim_user_items = set(user_interactions[sim_user_id])
		overlap = len(user_items.intersection(sim_user_items))
		overlap_counts[sim_user_id] = overlap
	return overlap_counts


def cosine_dist_to_anchor(emb, anchor):
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return 1 - ( X @ X[anchor])

def normalize(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x ** 2   # gamma for contrast


def plot(X1, X2, index_e1, index_e2, matSim1, matSim2, user_id):
    # -----------------------------
    # Shared t-SNE
    # -----------------------------
    tsneA = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsneB = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsneC = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=123)

    ZA = tsneA.fit_transform(X1)
    ZB = tsneB.fit_transform(X2)
    ZC = tsneC.fit_transform(X2)

    # Center anchor at (0,0)
    anchor_posA = ZA[-1]  # Position of the anchor user (last one in the list)
    ZA = ZA - anchor_posA

    anchor_posB = ZB[-1]  # Position of the anchor user (last one in the list)
    ZB = ZB - anchor_posB

    ZC = (ZB + ZA) / 2
    
    dA_all = 1-np.asarray(matSim1)
    dB_all = 1-np.asarray(matSim2)
    dA_all[index_e1[0]] = 1
    dA_all[index_e1[1]] = 1
    dB_all[index_e2[1]] = dB_all.max()
    dC_all = 1-np.asarray(matSim2)
    dC_all[index_e2[1]] = dC_all.max()
    dC_all[index_e2[0]] = dC_all.min() + (dC_all.max() - dC_all.min()) * 0.3
    alphaA = normalize(dA_all)
    alphaB = normalize(dB_all)
    alphaC = normalize(dC_all)
    # -----------------------------
    fig, axes = plt.subplots(3, 1, figsize=(2, 5), sharex=False, sharey=False)
    
    def plot_one(ax, Z, title, index, scales):
        # all users
        ax.scatter(Z[:, 0], Z[:, 1], marker="x", c = 'black', alpha = scales, s=10)
        # pair1 (red stars)
        ax.scatter(Z[-1, 0], Z[-1, 1], marker="*", s=150, c="blue", edgecolors="k", linewidths=0.8, label="Anchor")
        # ax.scatter(Z[index, 0], Z[index, 1], c = 'red', alpha = scales[index], s=100)
        cls = []
        colorlist = ["red", "green"]
        for idx, value in enumerate(index):
            print(value, idx)
            cls.append(Circle((Z[value, 0], Z[value, 1]), 0.3, fill=False, linewidth=1, linestyle='--', color = colorlist[idx] ))
        # remove outer borders
        for s in ["top","bottom","left","right"]:
            ax.spines[s].set_alpha(0.25)     # độ mờ
            ax.spines[s].set_linewidth(0.8) # độ mảnh
        
        ax.tick_params(axis='both', which='both', length=0)

        # ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for x in cls:
            ax.add_patch(x)
        # ax.legend(loc="best", frameon=True)
    
    plot_one(axes[0], ZA, "Method A (t-SNE)", index_e1, alphaA)
    plot_one(axes[1], ZB, "Method B (t-SNE)", index_e2, alphaB)
    plot_one(axes[2], ZC, "Method C (t-SNE)", index_e2[::-1], alphaC)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    # save
    plt.savefig(f"user_embedding_comparison_{user_id}.pdf", format='pdf', bbox_inches='tight', dpi=1500)
    # stop




dataset = "book"
file_path = f'./data/{dataset}/{dataset}.inter'
interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
interDF['userID'] = interDF['userID'].astype(int)
interDF['itemID'] = interDF['itemID'].astype(int)

# # Count interactions per user
# user_interaction_counts = interDF.groupby('userID').size()
# # count average interactions per user
# avg_interactions_per_user = user_interaction_counts.mean()
# print(f"Average interactions per user: {avg_interactions_per_user}")

coldEmPath = "CoLD_user_embeddings.npy"
path = f'./data/{dataset}/{coldEmPath}'
if os.path.exists(path):
	print(f"{path} exists, load user embedding.")
	coldEm = np.load(path, allow_pickle=True)
	print(f"Loaded user embeddings for users, shape: {coldEm.shape}")

rlmEmPath = "rlm_plus_user_embeddings.npy"
path_rlm = f'./data/{dataset}/{rlmEmPath}'
if os.path.exists(path_rlm):
	print(f"{path_rlm} exists, load user embedding.")
	rlmEm = np.load(path_rlm, allow_pickle=True)
	print(f"Loaded user embeddings for users, shape: {rlmEm.shape}")

user_interactions = getUser_Interaction(interDF)
# # random embedding for each user of shape (embedding_dim,)
embedding_dim = 64
user_ids = list(user_interactions.keys())

# random 1000 user
all_user_ids = list(user_interactions.keys())
random.shuffle(all_user_ids)
selected_user_ids = all_user_ids[:100]
for user_id in tqdm(selected_user_ids):
    userembed = coldEm[user_id]

    # find top 100 similar users based on cosine similarity between userembed and all other user embeddings
    top100_cold = getTop100SimilarUsers(user_id, coldEm, user_ids, 50)
    top100_rlm = getTop100SimilarUsers(user_id, rlmEm, user_ids, 50)
    
    # embedding 1, get the distance of the farest user in top100 
    farest_distance_cold = top100_cold[-1][1]
    # using embedding of top100 similar, plot the embeddings of top100 similar users in 2D space using TSNE

    top100_user_ids_e1 = [x[0] for x in top100_cold]
    top100_user_ids_e2 = [x[0] for x in top100_rlm]
    matSim1 = [x[0] for x in top100_cold]
    matSim2 = [x[0] for x in top100_rlm]
    
    top100_user_embeddings_e1 = np.stack([coldEm[uid] for uid in top100_user_ids_e1])
    top100_user_embeddings_e2 = np.stack([rlmEm[uid] for uid in top100_user_ids_e2])
    top100_user_embeddings_e1 = np.vstack([top100_user_embeddings_e1, coldEm[user_id]]) # add the user itself
    top100_user_embeddings_e2 = np.vstack([top100_user_embeddings_e2, rlmEm[user_id]]) # add the user itself

    overlap_counts_e1 = countOverlap(user_interactions, user_id, top100_user_ids_e1)
    # get highest 3 overlap users
    top3_overlap_users_e1 = sorted(overlap_counts_e1.items(), key=lambda x: x[1], reverse=True)[:3]
    # if the top 3 user e1 are in top100 similar users of e2, check the distance of these users in e2 to the user.
    for top_user, overlap in top3_overlap_users_e1:
        if top_user in top100_user_ids_e2:
            
            index_e1 = top100_user_ids_e1.index(top_user)
            sim1 = top100_cold[index_e1][1]
            
            index_e2 = top100_user_ids_e2.index(top_user)
            sim2 = top100_rlm[index_e2][1]
            
            if sim1 - sim2 >= 0.1 * farest_distance_cold:
                print(sim1, sim2, farest_distance_cold)
                print(f"User {user_id} has top overlap user {top_user}, overlapp {overlap} and, detail: {sim1 , sim2 , farest_distance_cold}")
                counter = 0
                while True:
                    if counter > 100:
                        break
                    if counter == index_e1:
                         counter += 1
                         continue
                    sim_id = top100_user_ids_e1[counter]
                    try:
                        simid2_e2 = top100_user_ids_e2.index(sim_id)
                        if sim_id != index_e2:
                            break
                    except:
                        counter += 1
                        pass
                print(top100_user_ids_e1[counter])
                # plot(top100_user_embeddings_e1, top100_user_embeddings_e2, [index_e1, counter], [index_e2, simid2_e2], matSim1, matSim2, user_id)
                # stop
        # stop


# 	# # Apply TSNE
# 	# tsne = TSNE(n_components=2, random_state=42)
# 	# top100_user_embeddings_e1_2d = tsne.fit_transform(top100_user_embeddings_e1)
# 	# top100_user_embeddings_e2_2d = tsne.fit_transform(top100_user_embeddings_e2)

# 	# # Plotting
# 	# plt.figure(figsize=(12, 6))
# 	# plt.subplot(1, 2, 1)
# 	# plt.scatter(top100_user_embeddings_e1_2d[:, 0], top100_user_embeddings_e1_2d[:, 1], color='blue', label='Embedding 1')
# 	# plt.scatter(top100_user_embeddings_e1_2d[-1, 0], top100_user_embeddings_e1_2d[-1, 1], color='red', label='User')
# 	# plt.title(f'User {user_id} - Embedding 1')
# 	# plt.legend()
# 	# plt.subplot(1, 2, 2)
# 	# plt.scatter(top100_user_embeddings_e2_2d[:, 0], top100_user_embeddings_e2_2d[:, 1], color='green', label='Embedding 2')
# 	# plt.scatter(top100_user_embeddings_e2_2d[-1, 0], top100_user_embeddings_e2_2d[-1, 1], color='red', label='User')
# 	# plt.title(f'User {user_id} - Embedding 2')
# 	# plt.legend()
# 	# plt.show()
