# Git 使用说明

## 一、核心概念（只需记住这三步）

```
你的电脑上的文件  →  git add（暂存）  →  git commit（存档）  →  git push（上传到GitHub）
```

| 命令 | 作用 | 比喻 |
|------|------|------|
| `git add` | 选择要上传的文件 | 把东西放进快递箱 |
| `git commit` | 保存一个版本记录 | 封箱并写上备注 |
| `git push` | 上传到 GitHub | 寄出快递 |

---

## 二、第一次配置（只做一次）

在终端（命令行）里输入以下命令，设置你的身份信息：

```bash
git config --global user.name "star666-jpg"
git config --global user.email "xuy170286@gmail.com"
```

**保存 GitHub 密码，避免每次输入：**

```bash
git config --global credential.helper store
```

> 第一次 push 时输入用户名和 token 后，之后会自动记住，不用再输。

---

## 三、日常使用流程

### 方式A：命令行（适合 WSL / 终端）

```bash
# 第一步：进入项目目录
cd /tmp/-LSTM-

# 第二步：查看哪些文件有修改
git status

# 第三步：暂存修改的文件（. 表示所有文件）
git add .

# 第四步：保存版本，引号里写本次修改的说明
git commit -m "说明这次改了什么"

# 第五步：上传到 GitHub
git push
```

---

### 方式B：VS Code（推荐，图形界面更直观）

**安装后首次绑定仓库：**

1. 打开 VS Code，点击菜单 `文件` → `打开文件夹`，选择项目所在文件夹
2. 点击左侧 **源代码管理** 图标（像树枝分叉的图标，快捷键 `Ctrl+Shift+G`）
3. 点击 `...` 菜单 → `远程` → `添加远程存储库`
4. 输入：`https://github.com/star666-jpg/-LSTM-.git`

**之后每次提交只需三步：**

1. 左侧源代码管理面板会显示所有修改的文件，点击文件旁边的 `+` 号暂存
2. 在上方输入框写备注（比如"添加了训练脚本"）
3. 点击 **提交** 按钮，再点击 **同步更改**（即 push）

---

### 方式C：PyCharm / JetBrains IDE

1. 打开项目后，菜单 `VCS` → `Enable Version Control Integration` → 选 `Git`
2. 菜单 `Git` → `Manage Remotes` → 点 `+` → 输入仓库地址
3. 之后修改文件后，按 `Ctrl+K` 提交，`Ctrl+Shift+K` 推送

---

## 四、第一次在新电脑上下载项目

```bash
git clone https://github.com/star666-jpg/-LSTM-.git
cd -LSTM-
```

---

## 五、常用命令速查

```bash
git status          # 查看当前有哪些文件被修改
git log --oneline   # 查看历史版本记录
git diff            # 查看具体改了什么内容
git pull            # 从GitHub拉取最新代码（多人协作时用）
```

---

## 六、Token 安全使用

**永远不要把 Token 粘贴到聊天软件或代码里。**

正确做法：
1. 在终端第一次 `git push` 时，提示输入密码，把 token 粘贴进去
2. 因为已配置 `credential.helper store`，之后会自动记住

如果 token 泄露，立即去 GitHub → Settings → Developer settings → Personal access tokens 撤销它。

---

## 七、出错了怎么办

| 错误提示 | 原因 | 解决方法 |
|---------|------|---------|
| `Permission denied` | token 没有权限 | 检查 token 是否有 `repo` 权限 |
| `rejected: non-fast-forward` | GitHub 上有你本地没有的内容 | 先 `git pull`，再 `git push` |
| `nothing to commit` | 没有新改动 | 正常，不需要操作 |
| `not a git repository` | 不在项目目录里 | 用 `cd` 进入正确目录 |
