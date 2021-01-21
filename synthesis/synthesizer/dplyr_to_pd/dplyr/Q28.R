df0 <- input1 %>% mutate(time_between = trending_date - publish_time) %>%
     group_by(time_between) %>%
     summarise(Sum=n()) %>%
     arrange(time_between) %>%
     top_n(10)
