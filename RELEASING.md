# 发布到 Maven Central

坐标：`io.github.9178nekomc:basalt:3.7.0`

Central 的规则比私服严：**每个 artifact 必须带 sources jar、javadoc jar 和一份脱离的 GPG 签名**，
POM 里必须有 license / developer / scm，命名空间必须验证归属。这些我已经在 `pom.xml` 里配好，
剩下的四件事只能你本人做（都需要账号和私钥）。

---

## 一次性准备

### 1. 注册 Central 账号并认领命名空间

1. 打开 <https://central.sonatype.com> → 用 GitHub 账号（`9178nekomc`）登录
2. 右上角 **Publish** → **Namespace** → 添加 `io.github.9178nekomc`
3. `io.github.<用户名>` 这种前缀由 Central 自动核验 GitHub 归属，通常几分钟内变成 `Verified`

> 这一步不过，后面所有上传都会被拒。命名空间一旦注册不可转让。

### 2. 生成上传 token

**Account** → **Generate User Token**，得到一对 `username` / `password`（形如 `AbCdEf...`，不是你的登录密码）。
写进 `~/.m2/settings.xml`：

```xml
<settings>
  <servers>
    <server>
      <id>central</id>
      <username>刚才生成的 username</username>
      <password>刚才生成的 password</password>
    </server>
  </servers>
</settings>
```

`<id>central</id>` 必须和 `pom.xml` 里 `<publishingServerId>central</publishingServerId>` 一致。

### 3. 生成 GPG 签名密钥

```bash
gpg --quick-gen-key "9178nekomc <你的邮箱>" ed25519 sign 0
gpg --list-secret-keys --keyid-format=long        # 记下 40 位 key id
gpg --keyserver keys.openpgp.org --send-keys <KEY_ID>
```

上传后去 <https://keys.openpgp.org> 点邮件里的确认链接，**不确认的话 Central 校验签名会失败**。

### 4. 确认本机能跑通打包

```bash
mvn -B clean package           # 不需要密钥，日常用
mvn -B -P release verify       # 会尝试签名，先确认 javadoc jar 能生成
```

---

## 正式发布

```bash
export MAVEN_GPG_PASSPHRASE='你的私钥口令'
mvn -B -P release deploy
```

会发生什么：

1. `maven-source-plugin` 挂上 `basalt-3.7.0-sources.jar`
2. `maven-javadoc-plugin` 生成 `basalt-3.7.0-javadoc.jar`
   （带 `--add-modules jdk.incubator.vector` 和 `-doclint none`，否则 Tensor 层会卡在缺 `@param`）
3. `maven-gpg-plugin` 对每个 artifact 生成 `.asc`
4. `central-publishing-maven-plugin` 把整包上传到 Central Portal，`autoPublish=true` 意味着
   校验通过后自动转成 `published`

全过程 5～20 分钟（要等 Central 自己校验）。状态看 <https://central.sonatype.com/publishing/deployments>。

发布成功后，别人就能这样用：

```xml
<dependency>
  <groupId>io.github.9178nekomc</groupId>
  <artifactId>basalt</artifactId>
  <version>3.7.0</version>
</dependency>
```

首次同步到 <https://repo1.maven.org/maven2/io/github/9178nekomc/basalt/> 大约再等 10～30 分钟。

---

## 常见失败

| 报错 | 原因 |
|---|---|
| `Namespace not registered` | 第 1 步没做完，或前缀不是 `io.github.<你的用户名>` |
| `Invalid signature` / `No public key` | 公钥没上传到 keys.openpgp.org，或邮箱没确认 |
| `Missing: javadoc` / `Missing: sources` | 没用 `-P release`，或 javadoc 插件被 `-Dmaven.javadoc.skip` 跳过 |
| `401 Unauthorized` | `settings.xml` 里用的是登录密码，不是 User Token |
| `version already exists` | Central 不可覆盖，改版本号重来 |

## 版本号约定

`3.7.0` 沿用 GitHub Release 的版本。因为这次同时改了 Java 包名（`jsl.*` → `basalt.*`）
和持久化魔数（`JSLM` → `BSLT`），**3.7.0 与任何旧版本都不兼容**——不过旧坐标从未发布过，
所以不构成真正的破坏性变更。下一次纯功能升级再从 `3.7.0` 走。
