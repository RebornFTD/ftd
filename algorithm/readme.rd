======================
搜索引擎中文分词算法（1）-----mmseg源码基础

>中文分词是自然语言处理的基础，但是由于中文的博大精深，一个句子在不同语境下可以进行不同划分，  加之又不具有英文书写天然空格隔开，所以无法形成标准， mmseg和swsc是目前比较大众的开源中文分词算法，分别用于coreseek（sphix）和xunsearch（xaplian）中，最近因为搜索需要，使用了coreseek， 但是需求比较特殊，有很多特殊情况并不理想，跟了一下源码，记录在此权当笔记。

**mmseg**分词算法的包含以下四条基本策略：

-  **组合长度最大**
-  **组合中平均词语长度最大**
-  **词语长度的变化率最小**
-  **计算组合中所有单字词词频的自然对数，然后将得到的值相加，取总和最大的词组**

具体不详细介绍，可以参考http://blog.csdn.net/pwlazy/article/details/17562927，本文主要分析一下mmseg分词源码，其实看到上面的 四条规则，会发现规则其实很容易理解，算法也不难，下面给出mmseg源码实现：

