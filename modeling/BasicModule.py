#coding:utf8
import torch as t
import time


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        partial = t.load(path)
        state = self.state_dict()
        pretrained_state = { k:v for k,v in partial.items() if k in state and v.size() == state[k].size() }
        state.update(pretrained_state)
        self.load_state_dict(state)

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
