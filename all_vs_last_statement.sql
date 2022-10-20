--Creating query for first test of last statement or every statement on target.
WITH BASE AS (
SELECT *
,ROW_NUMBER() OVER      (
                        PARTITION BY customer_id 
                        ORDER BY s_2
                        )
,ROW_NUMBER() OVER      (
                        PARTITION BY customer_id
                        ORDER BY s_2 DESC
                        ) last_statement_flag_drop
FROM TRAIN_DATA
)


SELECT *
,CASE WHEN last_statement_flag_drop = 1 then 1 else 0 end as last_statement_flag
,CASE WHEN (target = 1 AND last_statement_flag_drop = 1) then 1 else 0 end as last_statement_target
FROM BASE B
LEFT JOIN train_labels L
ON B.customer_id = L.customer_id
;