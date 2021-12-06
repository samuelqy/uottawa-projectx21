import os
import argparse
import cv2
import numpy as np
import face_recognition
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('--root')
parser.add_argument('--positive_ratio')
args = parser.parse_args()
root = args.root
positive_ratio = float(args.positive_ratio)

def image_num(root):
    users = os.listdir(root)
    image_num = 0
    for user in users:
        files = os.listdir(os.path.join(root, user))
        for file in files:
            if file[-4:] == '.jpg':
                image_num += 1
    print(f'root: {root}, image num: {image_num}, avg image num: {image_num / 1402}')

def tweet_num(root):
    users = os.listdir(root)

def pick_indicator_pic():
    i = 0
    pics = os.listdir('indicator_pic')
    positive_users = os.listdir('new_ds/positive')
    negative_users = os.listdir('new_ds/negative')
    for pic in pics:
        i += 1
        print(i)
        find = False

        j = 0
        for user in positive_users:
            j += 1
            print(f'positive {j} {i}')
            if find:
                break
            files = os.listdir(f'new_ds/positive/{user}')
            if pic[:-4] + '.jpg' in files:
                os.system(f'cp new_ds/positive/{user}/{pic[:-4]}.jpg selected_pic/positive/{pic[:-4]}.jpg')
                find = True
                break
            else:
                continue

        j = 0
        for user in negative_users:
            j += 1
            print(f'negative {j} {i}')
            if find:
                break
            files = os.listdir(f'new_ds/negative/{user}')
            if pic[:-4] + '.jpg' in files:
                os.system(f'cp new_ds/negative/{user}/{pic[:-4]}.jpg selected_pic/negative/{pic[:-4]}.jpg')
                find = True
                break
            else:
                continue

def remove_zero_pic():
    roots = ['positive', 'negative']
    for root in roots:
        pics = os.listdir(f'selected_pic/{root}')
        for pic in pics:
            if os.path.getsize(f'selected_pic/{root}/{pic}') == 0:
                os.remove(f'selected_pic/{root}/{pic}')

def RGB(root):
    image_num = 0.
    # selected = []
    # selected += os.listdir('selected_pic/positive')
    # selected += os.listdir('selected_pic/negative')
    users = os.listdir(root)
    h = np.zeros([180, 1])
    s = np.zeros([256, 1])
    v = np.zeros([256, 1])
    for user in users:
        print(user)
        image_paths = [path for path in os.listdir(os.path.join(root, user)) if path[-4:] == '.jpg']
        image_num += len(image_paths)
        for path in image_paths:
            path = os.path.join(root, user, path)
            img = cv2.imread(path)
            color = ('h', 's', 'v')
            for i, col in enumerate(color):
                if col == 'h':
                    histr = cv2.calcHist([img], [i], None, [180], [0, 180])
                else:
                    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                if col == 'h':
                    h += histr
                elif col == 's':
                    s += histr
                elif col == 'v':
                    v += histr
    h = np.divide(h, image_num).reshape(180)
    s = np.divide(s, image_num).reshape(256)
    v = np.divide(v, image_num).reshape(256)
    H, S, V = 0, 0, 0
    h_num, s_num, v_num = 0, 0, 0
    for i, _v in enumerate(h):
        H += i * _v
        h_num += _v
    H /= h_num
    for i, _v in enumerate(s):
        S += i * _v
        s_num += _v
    S /= s_num
    for i, _v in enumerate(v):
        V+= i * _v
        v_num += _v
    V /= v_num
    print(H)
    print(S)
    print(V)
    # r /= image_num
    # g /= image_num
    # b /= image_num
    # imgs = os.listdir(root)
    # for img in imgs:
    #     img = cv2.imread(os.path.join(root, img))
    #     color = ('b', 'g', 'r')
    #     for i, col in enumerate(color):
    #         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    #         if col == 'b':
    #             b += histr
    #         elif col == 'g':
    #             g += histr
    #         elif col == 'r':
    #             r += histr
    # r /= len(imgs)
    # g /= len(imgs)
    # b /= len(imgs)
    # with open('RGB/positive_r.pkl', 'rb') as f:
    #     r = pickle.load(f)
    # with open('RGB/positive_g.pkl', 'rb') as f:
    #     g = pickle.load(f)
    # with open('RGB/positive_b.pkl', 'rb') as f:
    #     b = pickle.load(f)
    #
    # data = {'r':r.reshape(256).tolist(), 'g':g.reshape(256).tolist(), 'b':b.reshape(256).tolist()}
    # df = pd.DataFrame(data)
    # writer = pd.ExcelWriter('positive.xlsx')
    # df.to_excel(writer)
    # writer.save()
    # plt.plot(r, color='r')
    # plt.plot(g, color='g')
    # plt.plot(b, color='b')
    # plt.xlim([0, 256])
    # plt.show()
    # with open(f'RGB/{prefix}_r.pkl', 'wb') as f:
    #     pickle.dump(r, f)
    # with open(f'RGB/{prefix}_g.pkl', 'wb') as f:
    #     pickle.dump(g, f)
    # with open(f'RGB/{prefix}_b.pkl', 'wb') as f:
    #     pickle.dump(b, f)
    H = 0
    S = 0
    V = 0
    img = cv2.imread('725143152022024192.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [1], None, [180], [0, 180])
    print(hist)

def face_reco(root):
    image_with_face_num = 0
    i = 0
    face_num = 0
    image_num = 0
    users = os.listdir(root)
    bar = progressbar.ProgressBar(max_value=len(users)).start()
    for user in users:
        i += 1
        bar.update(i)
        images = [image for image in os.listdir(os.path.join(root, user)) if image[-4:] == '.jpg']
        image_num += len(images)
        for image in images:
            img = face_recognition.load_image_file(os.path.join(root, user, image))
            face_locs = face_recognition.face_locations(img)
            if len(face_locs) > 0:
                image_with_face_num += 1
            face_num += len(face_locs)
    print(image_with_face_num / image_num)
    print(face_num / image_with_face_num)

def make_scale_ds(positive_ratio=0.1):
    positive_size = int(1500 * positive_ratio)
    negative_size = 1500 - positive_size
    all_positive_users = os.listdir('new_ds/positive')
    all_positive_users = np.array(all_positive_users)
    positive_index = np.random.choice(1401, positive_size, replace=False)
    positive_users = all_positive_users[positive_index]
    all_negative_users = os.listdir('new_ds/negative')
    all_negative_users = np.array(all_negative_users)
    negative_index = np.random.choice(1402, negative_size, replace=False)
    negative_users = all_negative_users[negative_index]
    for user in positive_users:
        os.system(f'cp -r new_ds/positive/{user} dataset/{positive_ratio}/positive/')
    for user in negative_users:
        os.system(f'cp -r new_ds/negative/{user} dataset/{positive_ratio}/negative/')

if __name__ == '__main__':
    make_scale_ds(positive_ratio=positive_ratio)