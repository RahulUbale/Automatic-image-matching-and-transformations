from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
import random
import sys
from PIL import Image
import numpy as np
import math
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as ch
import sys
import cv2
import glob
import operator

orb = cv2.ORB_create(nfeatures=1000)


def extract_features(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good.append([m])
    # img8 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # cv2.imwrite("img8.jpg", img8)
    # print(len(good))
    return len(good)


def dist(img1, img2):
    return 1


# print(sys.argv)

def pretty_print_matrix(matrix):
    for i in matrix:
        print(i)


def out_file_formatting(img_names, cluster_labels, out_file, k):
    with open(out_file, 'w') as f:
        for i in range(k):
            txt = ''
            for j in range(len(cluster_labels)):
                label = cluster_labels[j]
                if(label == i):
                    txt += img_names[j].split('/')[-1]
                    txt += ' '
            f.write(' '.join(txt))
            f.write('\n')
    pass


def transformation(src_im, trans_mat, fin_im, output_im):
    w = src_im.width
    h = src_im.height
    print(trans_mat)
    inv_trans_mat = np.linalg.inv(trans_mat)
    for m in range(w):
        for n in range(h):
            trans_res = np.array(inv_trans_mat).dot(np.array([m, n, 1]))
            # Here we are converting from 3D to 2D by eliminating the last element and dividing all elements by the
            # last element
            trans_res = (trans_res[:-1] / trans_res[-1])
            a = trans_res[0] - math.floor(trans_res[0])
            b = trans_res[1] - math.floor(trans_res[1])
            # if (trans_res[0]>0 and trans_res[0]< w) and (h > trans_res[1] > 0):
            if math.floor(trans_res[0]) > 0 and math.ceil(trans_res[0]) < w and math.floor(
                    trans_res[1]) > 0 and math.ceil(
                    trans_res[1]) < h:
                cord1 = np.array(src_im.getpixel((math.floor(trans_res[0]), math.floor(trans_res[1]))))
                cord2 = np.array(src_im.getpixel((math.ceil(trans_res[0]), math.floor(trans_res[1]))))
                cord3 = np.array(src_im.getpixel((math.ceil(trans_res[0]), math.ceil(trans_res[1]))))
                cord4 = np.array(src_im.getpixel((math.floor(trans_res[0]), math.ceil(trans_res[1]))))
                # if 0 < trans_res[0] < w and 0 < trans_res[1] < h:
                bil_int = (1 - a) * (1 - b) * cord1 \
                          + a * (1 - b) * cord2\
                          + a * b * cord3 \
                          + (1 - a) * b * cord4
                bil_int = tuple(bil_int.astype(int))
                fin_im.putpixel((m, n), (bil_int))

    fin_im.save(output_im)
    return fin_im

#part 1 driver code
def img_match(img_path, k, out_file):
    img_names = glob.glob(img_path)
    out_file = out_file
    k = k
    # print(len(img_names))
    n = len(img_names)
    matrix = [[0] * n for i in range(n)]

    for i_img1 in range(n):
        for i_img2 in range(i_img1 + 1, n):
            img1_path = img_names[i_img1]
            img2_path = img_names[i_img2]
            # print(img2_path)
            a = [img1_path]
            d = extract_features(img1_path, img2_path)

            matrix[i_img1][i_img2] = d
            matrix[i_img2][i_img1] = d

    # pretty_print_matrix(matrix)

    # dendrogram = ch.dendrogram(ch.linkage(matrix, method='ward'))
    cluster = AgglomerativeClustering(
        n_clusters=k, affinity="precomputed", linkage='single')
    cluster.fit_predict(matrix)
    # print(cluster.labels_)
    M = cluster.labels_

    # print(a)


    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=3, min_samples=2).fit(matrix)
    l = clustering.labels_
    # print(l)
    out_file_formatting(img_names, cluster.labels_, out_file, k)

    #other clutering approaches we tried:

    #dendrogram = ch.dendrogram(ch.linkage(matrix, method='ward'))


    # #Part 2 -1

    # cluster = AgglomerativeClustering(
    #     n_clusters=k, affinity="precomputed", linkage='single')
    # cluster.fit_predict(matrix)
    # print(cluster.labels_)
    # M = cluster.labels_

    # #Part 2 -2
    # clustering = DBSCAN(eps=3, min_samples=2).fit(matrix)
    # l = clustering.labels_
    # print(l)
    # #Part 2-3
    # sc = SpectralClustering(k, affinity='precomputed', n_init=100,
    #                         assign_labels='discretize')
    # sc.fit_predict(matrix)
    # K = sc.labels_
    # print(sc.labels_)


# #accuracy calculation function
# def part1_accuracy_calculation():

#     # Accuracy
#     ns = {x.replace('.jpg', '').replace(
#         'C:/Users/Rahul/Desktop/khusingh-dichalla-raubale-a2-main/part3-images\\', "").replace('_', '') for x in nm}

#     my_dict = dict(zip(ns, map(int, M)))
#     ns1 = list(ns)
#     print(ns1)
#     G = list(K)
#     N = len(G)
#     print(G)
#     TP = 0
#     TN = 0
#     ACC = 0
#     for i in range(N):
#         for j in range(i+1, N):

#             if ns1[i][:3] == ns1[j][:3] and G[i] == G[j]:
#                 TP += 1
#             elif ns1[i][:3] == ns1[j][:3] and G[i] != G[j]:

#                 TN += 1

#     ACC = (TN+TP)/(N*(N-1))
#     print(ACC)


#part2 driver code
def img_transform():
    img1 = sys.argv[3]
    img2 = sys.argv[4]
    src_img = Image.open(img1)
    des_img = Image.open(img2)
    output_img = sys.argv[5]
    src_siz = src_img.size
    fin_img = Image.new(mode="RGB", size=src_siz)
    # trans_mat = [[0.907, .0258, -182], [-0.153, 1.44, 58], [-0.000306, 0.000731, 1]]
    # new_img = transformation(img, trans_mat, fin_img, output_img)
    # return

    if int(sys.argv[2]) == 1:
        # translation
        if len(sys.argv) < 8:
            print("Check the total number of parameters")
            return
        src_cord = sys.argv[6]
        des_cord = sys.argv[7]
        src = [int(x) for x in src_cord.split(',')]
        des = [int(y) for y in des_cord.split(',')]
        trans_mat = [[1, 0, des[0] - src[0]], [0, 1, des[1] - src[1]], [0, 0, 1]]
        transformation(des_img, trans_mat, fin_img, output_img)
        return
    elif int(sys.argv[2]) == 2:
        # Euclidean
        src_cord1, des_cord1, src_cord2, des_cord2 = sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]
        src1, des1 = [int(x) for x in src_cord1.split(',')], [int(y) for y in des_cord1.split(',')]
        src2, des2 = [int(x) for x in src_cord2.split(',')], [int(y) for y in des_cord2.split(',')]
        if len(sys.argv) < 10:
            print("Check the total number of parameters")
            return
        src_mat = [[src1[0], -src1[1], 1, 0], [src1[1], src1[0], 0, 1],
                   [src2[0], -src2[1], 1, 0], [src2[1], src2[0], 0, 1]]
        des_mat = [[des1[0], des1[1], des2[0], des2[1]]]
        dest_mat = np.array(des_mat).T
        cos, sin, tx, ty = np.linalg.solve(src_mat, dest_mat)
        cos = float(cos)
        sin = float(sin)
        tx = float(tx)
        ty = float(ty)
        trans_mat = [[cos, -sin, tx], [sin, cos, ty], [0, 0, 1]]
        new_img = transformation(des_img, trans_mat, fin_img, output_img)
        return

    elif int(sys.argv[2]) == 3:
        # Affine transformation
        if len(sys.argv) < 12:
            print("Check the total number of parameters")
        src_cord1, des_cord1, src_cord2, des_cord2, src_cord3, des_cord3 = sys.argv[6], sys.argv[7], sys.argv[8], \
                                                                           sys.argv[9], sys.argv[10], sys.argv[11]
        src1, des1 = [int(x) for x in src_cord1.split(',')], [int(y) for y in des_cord1.split(',')]
        src2, des2 = [int(x) for x in src_cord2.split(',')], [int(y) for y in des_cord2.split(',')]
        src3, des3 = [int(x) for x in src_cord3.split(',')], [int(y) for y in des_cord3.split(',')]
        xy_mat = [[src1[0], src1[1], 1], [src2[0], src2[1], 1], [src3[0], src3[1], 1]]
        xdyd_mat = [[des1[0], des1[1], 1], [des2[0], des2[1], 1], [des3[0], des3[1], 1]]
        trans_mat = np.linalg.solve(xy_mat, xdyd_mat).T
        # trans_mat = trans_mat.flatten()
        new_img = transformation(des_img, trans_mat, fin_img, output_img)
        print("hihi")
        return

    elif int(sys.argv[2]) == 4:
        # Projective transformation
        if len(sys.argv) < 14:
            print("Check the total number of parameters")
            return
        # Projective transformation
        src_cord1, des_cord1, src_cord2, des_cord2 = sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]
        src_cord3, des_cord3, src_cord4, des_cord4 = sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13]
        src1, des1 = [int(x) for x in src_cord1.split(',')], [int(y) for y in des_cord1.split(',')]
        src2, des2 = [int(x) for x in src_cord2.split(',')], [int(y) for y in des_cord2.split(',')]
        src3, des3 = [int(x) for x in src_cord3.split(',')], [int(y) for y in des_cord3.split(',')]
        src4, des4 = [int(x) for x in src_cord4.split(',')], [int(y) for y in des_cord4.split(',')]
        pro_mat = [[src1[0], src1[1], 1, 0, 0, 0, -(src1[0] * des1[0]), -(src1[1] * des1[0])],
                   [0, 0, 0, src1[0], src1[1], 1, -(src1[0] * des1[1]), -(src1[1] * des1[1])],
                   [src2[0], src2[1], 1, 0, 0, 0, -(src2[0] * des2[0]), -(src2[1] * des2[0])],
                   [0, 0, 0, src2[0], src2[1], 1, -(src2[0] * des2[1]), -(src2[1] * des2[1])],
                   [src3[0], src3[1], 1, 0, 0, 0, -(src3[0] * des3[0]), -(src3[1] * des3[0])],
                   [0, 0, 0, src3[0], src3[1], 1, -(src3[0] * des3[1]), -(src3[1] * des3[1])],
                   [src4[0], src4[1], 1, 0, 0, 0, -(src4[0] * des4[0]), -(src4[1] * des4[0])],
                   [0, 0, 0, src4[0], src4[1], 1, -(src4[0] * des4[1]), -(src4[1] * des4[1])]]

        dest_mat = [[des1[0], des1[1], des2[0], des2[1], des3[0], des3[1], des4[0], des4[1]]]
        dest_mat = np.asarray(dest_mat).T
        print(len(dest_mat), len(pro_mat))
        h_mat = np.linalg.solve(pro_mat, dest_mat)
        h_mat = np.append(h_mat, [[1]], axis=0)
        trans_mat = np.reshape(h_mat, (3, 3))
        # apply transformation
        print("hi")
        new_img = transformation(des_img, trans_mat, fin_img, output_img)
        return
    else:
        print("wrong n")
        return


#part3
def extract_features_coordinates_part3(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    coordinates = []
    for m, n in matches:
      if m.distance < 0.75*n.distance:
        good.append([m])
        coordinates.append((
            (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])),
            (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1]))
        ))
    # img8 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # cv2.imwrite("img8.jpg", img8)
    # print(len(good))
    return coordinates


def projective_transformation_matrix_solver(pts_4):
  # Projective transformation
  src1, des1 = pts_4[0][0], pts_4[0][1]
  src2, des2 = pts_4[1][0], pts_4[1][1]
  src3, des3 = pts_4[2][0], pts_4[2][1]
  src4, des4 = pts_4[3][0], pts_4[3][1]

  pro_mat = [[src1[0], src1[1], 1, 0, 0, 0, -(src1[0] * des1[0]), -(src1[1] * des1[0])],
             [0, 0, 0, src1[0], src1[1], 1, -
                 (src1[0] * des1[1]), -(src1[1] * des1[1])],
             [src2[0], src2[1], 1, 0, 0, 0, -
                 (src2[0] * des2[0]), -(src2[1] * des2[0])],
             [0, 0, 0, src2[0], src2[1], 1, -
                 (src2[0] * des2[1]), -(src2[1] * des2[1])],
             [src3[0], src3[1], 1, 0, 0, 0, -
                 (src3[0] * des3[0]), -(src3[1] * des3[0])],
             [0, 0, 0, src3[0], src3[1], 1, -
                 (src3[0] * des3[1]), -(src3[1] * des3[1])],
             [src4[0], src4[1], 1, 0, 0, 0, -
                 (src4[0] * des4[0]), -(src4[1] * des4[0])],
             [0, 0, 0, src4[0], src4[1], 1, -(src4[0] * des4[1]), -(src4[1] * des4[1])]]

  dest_mat = [[des1[0], des1[1], des2[0], des2[1],
               des3[0], des3[1], des4[0], des4[1]]]
  dest_mat = np.asarray(dest_mat).T
  # print(len(dest_mat), len(pro_mat))

  #adding little noise to the matrix so that we don't get a singular matrix while computing inverse.
  pro_mat = pro_mat + 0.00001*np.random.rand(8, 8)
  dest_mat = dest_mat + 0.00001*np.random.rand(8, 1)
  h_mat = np.linalg.solve(pro_mat, dest_mat)
  h_mat = np.append(h_mat, [[1]], axis=0)
  trans_mat = np.reshape(h_mat, (3, 3))
  # # apply transformation
  # print("hi")
  return trans_mat

  # new_img = transformation(des_img, trans_mat, fin_img, output_img)


def transformation_part3(src_im, trans_mat):
    fin_im = Image.new(mode="RGB", size=src_im.size)
    w = src_im.width
    h = src_im.height
    print(trans_mat)
    inv_trans_mat = np.linalg.inv(trans_mat)
    for m in range(w):
        for n in range(h):
            trans_res = np.array(inv_trans_mat).dot(np.array([m, n, 1]))
            # Here we are converting from 3D to 2D by eliminating the last element and dividing all elements by the
            # last element
            trans_res = (trans_res[:-1] / trans_res[-1])
            a = trans_res[0] - math.floor(trans_res[0])
            b = trans_res[1] - math.floor(trans_res[1])
            # if (trans_res[0]>0 and trans_res[0]< w) and (h > trans_res[1] > 0):
            if math.floor(trans_res[0]) > 0 and math.ceil(trans_res[0]) < w and math.floor(
                    trans_res[1]) > 0 and math.ceil(
                    trans_res[1]) < h:
                cord1 = np.array(src_im.getpixel(
                    (math.floor(trans_res[0]), math.floor(trans_res[1]))))
                cord2 = np.array(src_im.getpixel(
                    (math.ceil(trans_res[0]), math.floor(trans_res[1]))))
                cord3 = np.array(src_im.getpixel(
                    (math.ceil(trans_res[0]), math.ceil(trans_res[1]))))
                cord4 = np.array(src_im.getpixel(
                    (math.floor(trans_res[0]), math.ceil(trans_res[1]))))
                # if 0 < trans_res[0] < w and 0 < trans_res[1] < h:
                bil_int = (1 - a) * (1 - b) * cord1 \
                    + a * (1 - b) * cord2\
                    + a * b * cord3 \
                    + (1 - a) * b * cord4
                bil_int = tuple(bil_int.astype(int))
                fin_im.putpixel((m, n), (bil_int))
    return fin_im


def part3(image1_path,image2_path,output_image_path):
    orb = cv2.ORB_create(nfeatures=1000)
    image1 = image1_path
    image2 = image2_path

    #getting the coordinates of match using orb
    coordinates = extract_features_coordinates_part3(image1, image2)
    print(coordinates)


    #RANSAC
    #performing RANSAC to find the best matching hyposis from the matches
    #By hypothesis I am considering the transformation matrix.
    hypothesis = []
    hypindx_votes = {}
    # creating an initial hypothesis
    trans_matrix = projective_transformation_matrix_solver(coordinates[:4])
    hypothesis.append(trans_matrix)
    hypindx_votes[0] = 0

    n_ransac = 64
    for i in range(n_ransac):
        samples_4 = random.sample(coordinates, 4)
        trans_matrix = projective_transformation_matrix_solver(samples_4)

    #check if current hypothesis matrix is similar to any matrix already present.
    #compute element wise difference and sum.
    for hyp_matrix_i in range(len(hypothesis)):
        if(np.sum(trans_matrix-hypothesis[hyp_matrix_i]) < 1):
            hypindx_votes[hyp_matrix_i] += 1
            break
        else:
            hypothesis.append(trans_matrix)
            hypindx_votes[len(hypothesis)-1] = 0    


    # Finally get the hypothesis with the maximum number of votes
    H = hypothesis[max(hypindx_votes.items(), key=operator.itemgetter(1))[0]]
    print(H)




    PIL_img1 = Image.open(image1)
    PIL_img2 = Image.open(image2)

    #transforming both images to same cordinate system by using the best transformation matrix computed.
    transformed_PIL_img2 = transformation_part3(PIL_img2, H)

    #creating a panorama by stiching the images together
    images = [PIL_img1, transformed_PIL_img2]
    widths, heights = zip(*(i.size for i in images))
    total_width, max_height = sum(widths), max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    
    new_im.save(output_image_path)


if __name__ == '__main__':
    print(sys.argv)

    if sys.argv[1] == 'part1':
        img_path = sys.argv[3:-1][0]
        out_file = sys.argv[-1]
        k = int(sys.argv[2])
        img_match(img_path, k, out_file)
        print("to be integrated")
    elif sys.argv[1] == 'part2':
        trans_mat = [[0.907, .0258, -182], [-0.153, 1.44, 58], [-0.000306, 0.000731, 1]]
        img = Image.open("lincoln.jpg")
        fin_img = Image.new(mode="RGB", size=img.size)
        out = "lin_output.png"
        transformation(img, trans_mat, fin_img, out)
        img_transform()
    elif sys.argv[1]=='part3':
        part3(image1_path= sys.argv[2],image2_path= sys.argv[3],output_image_path= sys.argv[4])
