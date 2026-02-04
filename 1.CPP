#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> vec = {1,2,3,4}; // 原内容：[1,2,3,4]，大小4
    vec.assign(3, 0);            // 清空原内容，填充5个0
    // 输出：0 0 0 0 0 ，大小：5
    for (int num : vec) cout << num << " ";
    cout << "\n大小：" << vec.size() << endl;
    return 0;
}