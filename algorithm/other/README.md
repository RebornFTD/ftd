####搜索引擎中文分词算法（1）-----mmseg源码基础

>中文分词是自然语言处理的基础，但是由于中文的博大精深，一个句子在不同语境下可以进行不同划分，  加之又不具有英文书写天然空格隔开，所以无法形成标准， mmseg和swsc是目前比较大众的开源中文分词算法，分别用于coreseek（sphix）和xunsearch（xaplian）中，最近因为搜索需要，使用了coreseek， 但是需求比较特殊，有很多特殊情况并不理想，跟了一下源码，记录在此权当笔记。

**mmseg**分词算法的包含以下四条基本策略：

-  **组合长度最大**
-  **组合中平均词语长度最大**
-  **词语长度的变化率最小**
-  **计算组合中所有单字词词频的自然对数，然后将得到的值相加，取总和最大的词组**

具体不详细介绍，可以参考http://blog.csdn.net/pwlazy/article/details/17562927，本文主要分析一下mmseg分词源码。

以上四条策略看起来很容易理解，然而最根本的问题是如何找到所有词语组合形式，再用规则来确定正确的分词方式，然而对于中文来讲有一些特殊的地方，首先计算机数据是以字节为单位，但是汉字却不是一个字节的，此外英文文章，单词之间都会以空格或者标点符号隔开，中文却不会，而且分词形式也会多种多样，这也正式中文分词的难点。

mmseg分词核心包括以下几个模块：
- **Chunk**  ：  元组
- **ChunkQueue**   :  元组队列
-  **MMThunk**  :  产生元组及元组队列
-  **Segmenter**  : 对输入文章句子进行词组匹配，
-  **其他模块**  ： UnigramCorpusReader（从文件中读取词典）、UnigramDict（提供基本几个接口）、ThesaurusDict（同义词典基本接口）

####1、元组
>元组（Chunk）：mmseg 以三个连续的分词为一个元组，最核心成员变量为vector<u2\> tokens，u2可以看成uint 主要表示的单词的长度，里面存储的是，连续三个或小于三个（当一句话由两个词组成，或者本身就是一个单词）的分词长度。

```
	class Chunk
	{
		inline void pushToken(u2 len, u2 freq)  
		inline float get_free()
		inline float get_avl() 
		inline float get_avg()
		inline void popup() 
		inline void reset() 
	
		float m_free_score;
		int total_length;
		std::vector<u2> tokens;
		std::vector<u2> freqs;			
	};
```
#### 2、 元组队列
> ChunckQueue :  元组队列中存储的是 一些列的元组，他们的起始位置都是一样的，举个例子比如 “长春市长春药店” 这句话分词，那么其可能的分词情况为：
> >长春市\_长春\_药店  `chunk: tokens<3, 2, 2>`
长春市\_长\_春药 `  chunk:tokens<3,1,2>  `      
长春\_市长\_春药`chunk:tokens<2,2,2>`
长春\_市\_长春`chunk:tokens<2,1,2>`
长\_春\_市长`chunk:tokens<1,1,2>`

>那么队列中 chunkQueue 的元组就为`[<3,2,2>, <3,1,2>,<2,2,2>,<2,1,2>,<1,1,2>]` 这些元组都是以这句话的开始“长”开始计算的，如果是很长的整篇文怎么办呢？（这个以后会继续讲），而chunkQueue中最重要的方法就是getToken了，其功能就是应用上面的四条规则去获取当前起始位置的一个分词：
```

	u2 getToken()
	{
		size_t num_chunk = m_chunks.size();
		if(!num_chunk)
				return 0;
			if(num_chunk == 1)
				return m_chunks[0].tokens[0];
			//debug use->dump chunk
	
			//do filter
			//apply rule 2
			float avg_length = 0;
			u4 remains[256]; //m_chunks.size can not larger than 256;
			u4* k_ptr = remains;
			for(size_t i = 0; i<m_chunks.size();i++){
				float avl = m_chunks[i].get_avl();
				if(avl > avg_length){
					avg_length = avl;
					k_ptr = remains;
					*k_ptr = (u4)i;
					k_ptr++;
				}else
				if(avl == avg_length){
					*k_ptr = (u4)i;
					k_ptr++;
				}
			}
			if((k_ptr - remains) == 1)
				return m_chunks[remains[0]].tokens[0]; //match by rule2
			//apply rule 3
			u4 remains_r3[256];
			u4* k_ptr_r3 = remains_r3;
			avg_length = 1024*64; //an unreachable avg 
			for(size_t i = 0; i<k_ptr-remains; i++){
				float avg = m_chunks[remains[i]].get_avg();
				if(avg < avg_length) {
					avg_length = avg;
					k_ptr_r3 = remains_r3;
					*k_ptr_r3 = (u4)remains[i];//*k_ptr_r3 = (u4)i;
					k_ptr_r3++;
				}else
				if(avg == avg_length){
					*k_ptr_r3 = (u4)i;
					k_ptr_r3++;
				}
			}
			if((k_ptr_r3 - remains_r3) == 1)
				return m_chunks[remains_r3[0]].tokens[0]; //match by rule3 min avg_length
			//apply r4 max freedom
			float max_score = 0.0;
			size_t idx = -1;
			for(size_t i = 0; i<k_ptr_r3-remains_r3; i++){
				float score = m_chunks[remains_r3[i]].get_free();
				if(score>max_score){
					max_score = score;
					idx = remains_r3[i];
				}
			}
			return m_chunks[idx].tokens[0];
			//return 0;
		};

		std::vector<Chunk> m_chunks;

```
```
	size_t commonPrefixSearch(const key_type *key,
							  u4 flag,
                              T* result,
                              size_t result_len,
                              size_t len = 0,
                              size_t node_pos = 0) {
      if (!len) len = length_func_()(key);

      register array_type_  b   = array_[node_pos].base;    //每次搜索 都会从 base=0 开始
	
	......//省略部分变量初始化代码
	
      for (register size_t i = 0; i < len; ++i) 
      {
        p = b;  // + 0;
        n = array_[p].base;
        if ((array_u_type_) b == array_[p].check && n < 0)                //已经批配到一个了，但是并没有结束，要找到从i=0开始所有可能的分词结果 
        {
          // result[num] = -n-1;
		  //found a sub word
          if (num < result_len) set_result(result[num], -n-1, i , p);    //找到一个分词，但是不会停止，继续直到匹配到最长的词        
          ++num;
        }
        p = b +(node_u_type_)(key[i]&flag) + 1;                         //注意，匹配下一个字节
           
        if ((array_u_type_) b == array_[p].check)                       //检查，是否满足条件
          b = array_[p].base;
		else                                                            //不满足，匹配结束终止查询
		{
		  //found a mismatch
          return num;
		}
      }
```

分词模块 **MMThunk** 主要包括以下接口：
-  **pushChunk**
-  **Tokenize**
-  **setItems**
```
int MMThunk::Tokenize()
{

	// appply rules
	u2 base = 0;
	while(base<=m_max_length)
	{
		Chunk chunk;
		item_info* info_1st = m_charinfos[base];
		for(size_t i = 0; i<info_1st->items.size(); i++)
		{
			if(i == 0)
				chunk.pushToken(info_1st->items[i], info_1st->freq);
			else
				chunk.pushToken(info_1st->items[i],0);
			//Chunk L1_chunk = chunk;
			u2 idx_2nd =  info_1st->items[i] + base;
			//check bound
			item_info* info_2nd = NULL;
			if(idx_2nd<m_max_length)
				info_2nd = m_charinfos[idx_2nd];
			if(info_2nd)
			{
				for(size_t j = 0; j<info_2nd->items.size(); j++) 
				{
					if(j == 0)
						chunk.pushToken(info_2nd->items[j], info_2nd->freq);
					else
						chunk.pushToken(info_2nd->items[j],1);
					u2 idx_3rd = info_2nd->items[j] + idx_2nd;
					if(idx_3rd<m_max_length && m_charinfos[idx_3rd]) 
					{
						u2 idx_4th = m_charinfos[idx_3rd]->items[m_charinfos[idx_3rd]->items.size()-1];
						if(m_charinfos[idx_3rd]->items.size() == 1)
							chunk.pushToken(idx_4th, m_charinfos[idx_3rd]->freq );
						else
							chunk.pushToken(idx_4th, 1);
						//push path.
						pushChunk(chunk);
						//pop 3part
						chunk.popup();
					}
					else
					{
						//no 3part, push path
						pushChunk(chunk);
					}
					//pop 2part
					chunk.popup();
				}//end for
			}//end if
			else
			{
				//no 2part ,push path
				pushChunk(chunk);
			}
			//pop 1part
			chunk.popup();
		}
		//find the last pharse
		//reset. rebase
		u2 tok_len = m_queue.getToken();
		if(tok_len)
		{
			pushToken(tok_len, base); //tokens.push_back(tok_len);
		}else
			break;
		m_queue.reset();
        chunk.reset();
		base += tok_len;
	}//end while
	return 0;
}
```
