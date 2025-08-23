def modify_list(lst):
    lst.append(4)      # Thay đổi nội dung danh sách (OK, ảnh hưởng ra ngoài)
    lst = [1, 2, 3]    # Gán lst trỏ sang list mới (KHÔNG ảnh hưởng ra ngoài)

my_list = [0]
modify_list(my_list)
print(my_list)
