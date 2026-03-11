public class reverse {
    static void reverseArray(int arra[], int sz) {
        int start = 0;
        int end = sz - 1;

        while (start < end) {
            int temp = arra[start];
            arra[start] = arra[end];
            arra[end] = temp;

            start++;
            end--;
        }
    }
    public static void main(String args[]) {
        int arra[] = { 20, 30, 40, 50, 60, 70 };
        int sz = arra.length;
        reverseArray(arra, sz);
        for (int i = 0; i < sz; i++) {
            System.out.println(arra[i]);
        }
    }
}
