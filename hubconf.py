dependencies = ['paddle']

import paddle
from model import MM as _MM


def MM(pretrained=False):
    '''This is a test demo for paddle hub
    '''
    mm = _MM()
    if pretrained:
        url = 'https://github.com/lyuwenyu/paddlehub_demo/releases/download/v1.0/params.pd'
        path = paddle.utils.download.get_weights_path_from_url(url)
        mm.set_state_dict(paddle.load(path))
    return mm
    
