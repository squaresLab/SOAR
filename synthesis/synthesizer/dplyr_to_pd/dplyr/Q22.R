df0 <- input1 %>% filter(Year == 2017) %>%
			group_by(Nationality) %>% summarise(n=sum(Number)) %>%
			top_n(10, n) %>% arrange(desc(n))