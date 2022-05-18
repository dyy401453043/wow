from parlai.agents.rag.retrievers import retriever_factory
from parlai.agents.rag.args import setup_rag_args
from parlai.core.script import ParlaiScript
from parlai.core.params import ParlaiParser
from parlai.agents.rag.rag import RagAgent
import torch
from tqdm import tqdm
import json

class Retrieval(ParlaiScript):
    @classmethod
    def setup_args(cls):
        """
        File in/out args, and sharding args.
        """
        parser = ParlaiParser(True, True, 'duyiyang')
        parser.add_argument(
            '--path_to_index',
            type=str,
            help='',
            default='/data/duyiyang1/ParlAI-main/data/models/hallucination/wiki_index_compressed/compressed_pq'
        )
        parser.add_argument(
            '--path_to_dpr_passages',
            type=str,
            help='',
            default='/data/duyiyang1/ParlAI-main/data/models/hallucination/wiki_passages/psgs_w100.tsv'
        )
        parser.add_argument(
            '--rag_retriever_type', type=str, help='retriever_type',default='dpr'
        )
        parser.add_argument(
            '--fp16', type=str, help='retriever_type', default=10
        )
        parser.add_argument('--no_cuda', type=bool, default='True')
        parser = setup_rag_args(parser)
        return parser


    def run(self, query_str="I like Gardening, even when I've only been doing it for a short time.[SEP]"):
        print(self.opt)
        fid_data_path = './test_wow.json'
        output_path = './test_wow_50.json'
        print(f'load data from {fid_data_path}')
        data = json.load(open(fid_data_path,'r'))
        retriever = retriever_factory(self.opt, RagAgent.dictionary_class()(self.opt))
        for item in tqdm(data):
            query = retriever.tokenize_query(item['question'])
            docs, scores = retriever.retrieve_and_score(torch.tensor([query]))
            item['ctxs'] = [{'title':doc.get_title(), 'text':doc.get_text()} for doc in docs[0]]
        print(f'write data to {output_path}')
        json.dump(data, open(output_path,'w'), indent=4)


if __name__ == '__main__':
    Retrieval.main()
