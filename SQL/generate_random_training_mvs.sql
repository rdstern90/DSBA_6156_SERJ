create materialized view train_labels_random as
select *
from train_labels_all
order by random()
limit 10000;

create materialized view train_data_random as
select train_data_all.*
from train_data_all
inner join train_labels_random
on train_data_all.customer_id = train_labels_random.customer_id;