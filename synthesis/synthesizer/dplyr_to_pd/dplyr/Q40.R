df <- input1 %>% group_by(Gender) %>% summarise(n=sum(JobSatisfaction))
