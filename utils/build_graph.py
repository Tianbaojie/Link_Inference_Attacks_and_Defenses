
import dgl
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

import copy
def build_graph_dgl(Users,Questions,Answers_dev,Answers_test):
    user_following_user=[]
    for u,f in zip(Users.uid,Users.followings):
        for v in f:
            user_following_user.append([u,v])
    user_follower_user=[]
    for u,f in zip(Users.uid,Users.followers):
        for v in f:
            user_follower_user.append([u,v])
    user_answer_write=[]
    for u,f in zip(Users.uid,Users.writtenAnswers):
        try:
            for v in f:
                user_answer_write.append([u,v])
        except:
            pass
    user_following_topic=[]
    for u,f in zip(Users.uid,Users.followedTopics):
        try:
            for v in f:
                user_following_topic.append([u,v])
        except:
            pass
    user_ask_question=[]
    for u,f in zip(Users.uid,Users.askedQuestions):
        try:
            for v in f:
                user_ask_question.append([u,v])
        except:
            pass
    user_follow_question=[]
    for u,f in zip(Users.uid,Users.followedQuestions):
        try:
            for v in f:
                user_follow_question.append([u,v])
        except:
            pass

    question_belongto_topic=[]
    for u,f in zip(Questions.questionId,Questions.questionTopics):
        try:
            for v in f:
                question_belongto_topic.append([u,v])
        except:
            pass
    question_have_answer=[]
    for u,f in zip(Questions.questionId,Questions.answers):
        try:
            for v in f:
                question_have_answer.append([u,v])
        except:
            pass
    data={
    ('user', 'user_write_answer', 'answer') :(torch.tensor(user_answer_write)[:,0],torch.tensor(user_answer_write)[:,1]),
    ('answer', 'answer_write_user', 'user') :(torch.tensor(user_answer_write)[:,1],torch.tensor(user_answer_write)[:,0]),
    ("user", "user_follower_user", "user"):(torch.tensor(user_follower_user)[:,0],torch.tensor(user_follower_user)[:,1]),
    ("user", "user_following_user", "user"):(torch.tensor(user_following_user)[:,0],torch.tensor(user_following_user)[:,1]),
    ("user", "user_following_topic", "topic"):(torch.tensor(user_following_topic)[:,0],torch.tensor(user_following_topic)[:,1]),
    ("topic", "topic_following_user", "user"):(torch.tensor(user_following_topic)[:,1],torch.tensor(user_following_topic)[:,0]),
    ("user", "user_following_question", "question"):(torch.tensor(user_follow_question)[:,0],torch.tensor(user_follow_question)[:,1]),
    ("question", "question_following_user", "user"):(torch.tensor(user_follow_question)[:,1],torch.tensor(user_follow_question)[:,0]),
    ("user", "user_ask_question", "question"):(torch.tensor(user_ask_question)[:,0],torch.tensor(user_ask_question)[:,1]),
    ("question", "question_ask_user", "user"):(torch.tensor(user_ask_question)[:,1],torch.tensor(user_ask_question)[:,0]),
    ("question", "question_belongto_topic", "topic"):(torch.tensor(question_belongto_topic)[:,0],torch.tensor(question_belongto_topic)[:,1]),
    ("topic", "topic_belongto_question", "question"):(torch.tensor(question_belongto_topic)[:,1],torch.tensor(question_belongto_topic)[:,0]),
    ("question", "question_have_answer", "answer"):(torch.tensor(question_have_answer)[:,0],torch.tensor(question_have_answer)[:,1]),
    ("answer", "answer_have_question", "question"):(torch.tensor(question_have_answer)[:,1],torch.tensor(question_have_answer)[:,0]),
    }
    g=dgl.heterograph(data)
    return g
def build_graph_pyg(Users,Questions,Answers_train,Answers_dev,Answers_test):
    user_following_user=[]
    for u,f in zip(Users.uid,Users.followings):
        for v in f:
            user_following_user.append([u,v])
    user_follower_user=[]
    for u,f in zip(Users.uid,Users.followers):
        for v in f:
            user_follower_user.append([u,v])
    user_answer_write=[]
    for u,f in zip(Users.uid,Users.writtenAnswers):
        try:
            for v in f:
                user_answer_write.append([u,v])
        except:
            pass
    user_following_topic=[]
    Topics=[]
    for u,f in zip(Users.uid,Users.followedTopics):
        try:
            for v in f:
                user_following_topic.append([u,v])
                Topics.append(v)
        except:
            pass
    user_ask_question=[]
    for u,f in zip(Users.uid,Users.askedQuestions):
        try:
            for v in f:
                user_ask_question.append([u,v])
        except:
            pass
    user_follow_question=[]
    for u,f in zip(Users.uid,Users.followedQuestions):
        try:
            for v in f:
                user_follow_question.append([u,v])
        except:
            pass
    question_belongto_topic=[]
    for u,f in zip(Questions.questionId,Questions.questionTopics):
        try:
            for v in f:
                question_belongto_topic.append([u,v])
                Topics.append(v)
        except:
            pass
    question_have_answer=[]
    for u,f in zip(Questions.questionId,Questions.answers):
        try:
            for v in f:
                question_have_answer.append([u,v])
        except:
            pass
    data = HeteroData()
    data["user"].node_id = torch.arange(len(Users))
    data["question"].node_id = torch.arange(len(Questions))

    data["answer"].node_id = torch.arange(len(Answers_train)+2000)
    Topics=set(Topics)
    data["topic"].node_id = torch.arange(len(Topics))
    # Add the node features and edge indices:
    data["user", "write", "answer"].edge_index =torch.tensor(np.array(user_answer_write).T.tolist())
    data["user", "follower", "user"].edge_index=torch.tensor(np.array(user_follower_user).T.tolist())
    data["user", "following", "user"].edge_index=torch.tensor(np.array(user_following_user).T.tolist())
    data["user", "following", "topic"].edge_index=torch.tensor(np.array(user_following_topic).T.tolist())
    data["user", "following", "question"].edge_index=torch.tensor(np.array(user_follow_question).T.tolist())
    data["user", "ask", "question"].edge_index=torch.tensor(np.array(user_ask_question).T.tolist())
    data["question", "belongto", "topic"].edge_index=torch.tensor(np.array(question_belongto_topic).T.tolist())
    data["question", "have", "answer"].edge_index=torch.tensor(np.array(question_have_answer).T.tolist())
    data = T.ToUndirected()(data)

    """
    data_null = HeteroData({("user", "write", "answer"):{'edge_index':torch.tensor([[0,0]]).T},
                            ("user", "follower", "user"):{'edge_index':torch.tensor([[0,0]]).T},
                            ("user", "following", "user"):{'edge_index':torch.tensor([[0,0]]).T},
                            ("user", "following", "topic"):{'edge_index':torch.tensor([[0,0]]).T},
                            ("user", "following", "question"):{'edge_index':torch.tensor([[0,0]]).T},
                            ("user", "ask", "question"):{'edge_index':torch.tensor([[0,0]]).T},
                            ("question", "belongto", "topic"):{'edge_index':torch.tensor([[0,0]]).T},
                            ("question", "have", "answer"):{'edge_index':torch.tensor([[0,0]]).T},
                            })
    """
    data_null = HeteroData()
    data_null["user"].node_id = torch.arange(len(Users))
    data_null["question"].node_id = torch.arange(len(Questions))

    data_null["answer"].node_id = torch.arange(len(Answers_train)+2000)
    data_null["topic"].node_id = torch.arange(len(Topics))

    data_train_neg=copy.deepcopy(data_null)
    user_answer_write=[[[i,edge[1]] for i in np.random.randint(0,len(Users),4)] for edge in user_answer_write]
    user_answer_write=np.array(user_answer_write).reshape(-1,2).tolist()
    data_train_neg["user", "write", "answer"].edge_index =torch.tensor(np.array(user_answer_write).T.tolist())
    data_train_neg = T.ToUndirected()(data_train_neg)


    data_dev_pos=copy.deepcopy(data_null)
    user_answer_write=Answers_dev.apply(lambda x:[x['users'][0],x['answerId']],axis=1).tolist()
    data_dev_pos["user", "write", "answer"].edge_index =torch.tensor(np.array(user_answer_write).T.tolist())
    data_dev_pos = T.ToUndirected()(data_dev_pos)

    data_test_pos=copy.deepcopy(data_null)
    user_answer_write=Answers_test.apply(lambda x:[x['users'][0],x['answerId']],axis=1).tolist()
    data_test_pos["user", "write", "answer"].edge_index =torch.tensor(np.array(user_answer_write).T.tolist())
    data_test_pos = T.ToUndirected()(data_test_pos)

    data_dev_neg=copy.deepcopy(data_null)
    user_answer_write=Answers_dev.apply(lambda x:[[u,x['answerId']] for u in x['users'][1:]],axis=1).tolist()
    user_answer_write=np.array(user_answer_write).reshape(-1,2).tolist()
    data_dev_neg["user", "write", "answer"].edge_index =torch.tensor(np.array(user_answer_write).T.tolist())
    data_dev_neg = T.ToUndirected()(data_dev_neg)

    data_test_neg=copy.deepcopy(data_null)
    user_answer_write=Answers_test.apply(lambda x:[[u,x['answerId']] for u in x['users'][1:]],axis=1).tolist()
    user_answer_write=np.array(user_answer_write).reshape(-1,2).tolist()
    data_test_neg["user", "write", "answer"].edge_index =torch.tensor(np.array(user_answer_write).T.tolist())
    data_test_neg = T.ToUndirected()(data_test_neg)

    return data,data_train_neg,data_dev_pos,data_dev_neg,data_test_pos,data_test_neg
    
def build_meta_path_one(Users):
    meta_paths_one=[]
    for u,topics in zip(Users.uid,Users.followedTopics):
        try:
            meta_paths_one+=[[u,t] for t in topics]
        except:
            pass
    meta_paths_one=pd.DataFrame(meta_paths_one,columns=['uid','topic'])
    return meta_paths_one
def build_meta_path_two(Users,Questions):
    meta_paths_two=[]
    for u,questions in zip(Users.uid,Users.followedQuestions):
        try:
            for questionId in questions:
                try:
                    meta_paths_two+=[[u,t] for t in Questions[Questions['questionId']==questionId].questionTopics.values[0]]
                except:
                    pass
        except:
            pass
    return pd.DataFrame(meta_paths_two,columns=['uid','topic'])

def build_meta_path_three(Users,Questions):
    meta_paths_three=[]
    for u,questions in zip(Users.uid,Users.askedQuestions):
        try:
            for questionId in questions:
                try:
                    meta_paths_three+=[[u,t] for t in Questions[Questions['questionId']==questionId].questionTopics.values[0]]
                except:
                    pass
        except:
            pass
    return pd.DataFrame(meta_paths_three,columns=['uid','topic'])