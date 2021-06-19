# botnetGenerator

This code generates dataset for botnet detection with CAIDA background traffic and synthesized botnet. 
A CAIDA account is required before running this code.
You can refer to ``botnetGenerator.py`` to run the code. The basic usage could be
```
python3 botnetGenerator.py \
--CAIDA_user xx \
--CAIDA_password xx \
--CAIDA_link xx \
--dst_dir xx \
--dst_name xx \
--graph_id xx \
--start_time xx \
--stop_time xx \
--botnet_type xx \ 
--num_edge xx \
--num_node xx 
```
