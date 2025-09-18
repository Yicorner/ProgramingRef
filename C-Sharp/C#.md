## 内建反射
```C#
using System;
class Player {
    public int health = 100;
}

var p = new Player();
Type t = p.GetType();              // 获取类型信息
Console.WriteLine(t.Name);         // Player
var field = t.GetField("health");  // 获取字段信息
Console.WriteLine(field.GetValue(p)); // 100

```
## 属性系统
![](20250811155045.png)

## Unity3d脚本从唤醒到销毁有着一套比较完整的生命周期
Awake —> OnEnable —> Start —> FixedUpdate —>Update —> LateUpdate—> OnGUl —> OnDisable —> OnDestroy
![](20250811171231.png)
- 对象（Object） 指的是场景中所有已经被加载并实例化的组件和 GameObject
- OnEnable作用
![](20250811171602.png)
- 对象被禁用
![](20250811171702.png)
- 脚本实例被启用 
每个挂在 GameObject 上的脚本（继承 MonoBehaviour）本身也有一个 启用状态，由 MonoBehaviour.enabled 控制
- FixedUpdate里的固定帧 
![](20250811171923.png)
- update和lateupdate
![](20250811171958.png)
- 如果Monobehaviour的enabled属性设为false，OnGUI()将不会被调用，这里的 enabled 就是和刚才决定 Start() 是否会被调用的 同一个 MonoBehaviour.enabled 属性。
- OnDisable
![](20250811172137.png)
- 协同程序（Coroutine）
![](20250811172211.png)
- 对象“不可用”和“非激活状态”的区别
![](20250811172235.png)
- 其他组件
![](20250811172300.png)
## 生命周期主线程
Unity 的协程和 Update 这些生命周期函数，本质上都运行在主线程，并且是顺序调度的。
## 碰撞器
![](20250811195626.png)
## 触发器
![](20250811195638.png)
## 物体发生碰撞的必要条件
下图说的碰撞器是 is Trigger = false的Collider
![](20250811200008.png)
## .Net与Mono的关系
见C++.md
PS：Mono运行时即使CLR（虚拟机，负责执行编译后的IL）
## 生命周期
指一个 GameObject的生命周期的“起点”和“终点”
起点：对象被创建（Instantiate / 场景加载 / 编辑器放置）
终点：对象被销毁（Destroy / 场景卸载 / 应用退出）
## Baked Light
![](20250811201354.png)
## CharacterController和Rigidbody的区别
![](20250811201755.png)
## prefab 蓝图
![](20250811201935.png)
## 协程
![](202508112052133.png)
## 协程中的WaitForSeconds是否另一个线程
![](20250811210229.png)
## Invoke与Coroutine
Invoke：当前代码继续执行，不会在这里停下来
Coroutine：反之
## 正在运行的脚本，隐藏物体与禁止脚本导致触发OnDisable时，Invoke与coroutine是否正常运行？
![](20250812222813.png)
## 物体发生碰撞的整个过程
OnCollisionEnter、 OnCollisionStay、 OnCollisionExit三个函数
## Unity3d的物理引擎中，有几种施加力的方式
![](20250812223019.png)
## 对象旋转
自身旋转：transform.Rotate()
绕某点旋转：transform.RotateAround
## Image和RawImage
Imgae比RawImage更消耗性能
Image只能使用Sprite属性的图片，但是RawImage什么样的都可以使用
Image适合放一些有操作的图片，裁剪平铺旋转什么的，针对Image Type属性
RawImage就放单独展示的图片就可以，性能会比Image好很多
## 相机设置
![](20250813095116.png)
## 游戏动画类型
![](20250813095315.png)
## 如何让已经存在的GameObject在LoadLevel后不被卸载掉？
DontDestroyOnLoad(transform.gameObject);
## LightMap
LightMap：就是指在三维软件里实现打好光，然后渲染把场景各表面的光照输出到贴图上，最后又通过引擎贴到场景上，这样就使物体有了光照的感觉。
## Unity的着色器（渲染）
![](20250813101711.png)
## alpha test
![](20250813102752.png)
## 链条关节
Hinge Joint，可以模拟两个物体间用一根链条连接在一起的情况，能保持两个物体在一个固定距离内部相互移动而不产生作用力，但是达到固定距离后就会产生拉力。
## 游戏对象static
设置游戏对象为Static将会剔除（或禁用）网格对象当这些部分被静态物体挡住而不可见时。因此，在你的场景中的所有不会动的物体都应该标记为Static。
## 有A和B两组物体，有什么办法能够保证A组物体永远比B组物体先渲染？
把A组物体的渲染对列大于B物体的渲染队列
## 将图片的TextureType选项分别选为Texture和Sprite有什么区别
![](20250813104439.png)
## Terrain 分别贴3张，4张，5张地表贴图，渲染速度有什么区别？为什么？
Terrain = 用高度图 + 多张地表纹理 + 植被系统快速生成的大型户外地形
![](20250813104934.png)
## Dynamic Batching VS Static Batching
![](20250813105423.png)
## 什么是 Cookie（光照遮罩）
![](20250813110042.png)
![](20250813110051.png)
## Addcomponent后哪个生命周期函数会被调用
对于AddComponent添加的脚本，其Awake，Start，OnEnable是在Add的当前帧被调用的
其中Awake，OnEnable与AddComponent处于同一调用链上
Start会在当前帧稍晚一些的时候被调用，Update则是根据Add调用时机决定何时调用：如果Add是在当前帧的Update前调用，那么新脚本的Update也会在当前帧被调用，否则会被延迟到下一帧调用。
## 层（Layer）
![](20250813110827.png)

# View
## 如何在Unity3D中查看场景的面数，顶点数和Draw Call数
在Game视图右上角点击Stats。降低Draw Call 的技术是Draw Call Batching
