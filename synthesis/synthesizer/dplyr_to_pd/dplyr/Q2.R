df0 <- input1 %>% group_by(company) %>% summarise(count = n())
df1 <- df0 %>% top_n(n=10, wt=count)