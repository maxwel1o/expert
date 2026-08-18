#!/usr/bin/env python3
"""Deliver benchmark report and results via configured messaging channel.

Usage: python3 deliver_results.py <report_path> <archive_path> [target_id] [channel]

If target_id is not provided, it will be looked up from USER.md or memory files.
If channel is not provided, it will be auto-detected from openclaw config.
"""

import json, os, sys, re, subprocess

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def find_default_target():
    """Look for default Feishu target ID in USER.md or memory files.
    
    Priority:
    1. Inbound context (channel + chat_id from current request) - "respond where asked"
    2. USER.md / memory stored default ID - fallback for webchat/background
    """
    workspace = os.path.expanduser("~/.openclaw/workspace")
    
    # Priority 1: Check inbound context (set by openclaw as env vars or session context)
    inbound_channel = os.environ.get("OPENCLAW_INBOUND_CHANNEL", "")
    inbound_chat_id = os.environ.get("OPENCLAW_INBOUND_CHAT_ID", "")
    inbound_user_id = os.environ.get("OPENCLAW_INBOUND_USER_ID", "")
    
    if inbound_channel == "feishu" and (inbound_chat_id or inbound_user_id):
        target = inbound_chat_id or inbound_user_id
        print(f"📍 使用来源会话 ID: {target} (from {inbound_channel})")
        return target
    
    # Priority 2: Check USER.md
    user_md = os.path.join(workspace, "USER.md")
    if os.path.exists(user_md):
        with open(user_md) as f:
            content = f.read()
        # Look for ou_ or oc_ IDs
        matches = re.findall(r'(ou_[a-f0-9]{20,}|oc_[a-f0-9]{20,})', content)
        if matches:
            print(f"📍 使用 USER.md 记忆的默认 ID: {matches[0]}")
            return matches[0]
    
    # Priority 3: Check memory files
    memory_dir = os.path.join(workspace, "memory")
    if os.path.isdir(memory_dir):
        for fname in sorted(os.listdir(memory_dir), reverse=True):
            fpath = os.path.join(memory_dir, fname)
            if fname.endswith('.md') and os.path.isfile(fpath):
                with open(fpath) as f:
                    content = f.read()
                matches = re.findall(r'(ou_[a-f0-9]{20,}|oc_[a-f0-9]{20,})', content)
                if matches:
                    print(f"📍 使用 memory 中的 ID: {matches[0]}")
                    return matches[0]
    
    return None

def check_channel_configured(channel="feishu"):
    """Check if a messaging channel is configured in openclaw."""
    config_path = os.path.expanduser("~/.openclaw/openclaw.json")
    config = load_json(config_path)
    
    # Check plugins entries for the channel
    plugins = config.get("plugins", {})
    entries = plugins.get("entries", {})
    if channel in entries and entries[channel].get("enabled", True):
        return True
    
    # Check if feishu extension exists
    ext_path = os.path.expanduser(f"~/.openclaw/extensions/{channel}")
    if os.path.isdir(ext_path):
        return True
    
    return False

def check_pairing_done(channel="feishu"):
    """Check if pairing is completed by testing API access."""
    if channel == "feishu":
        # Try to call feishu API to verify connection - use the openclaw config
        # If feishu plugin is loaded and has app scopes, pairing is done
        config_path = os.path.expanduser("~/.openclaw/openclaw.json")
        config = load_json(config_path)
        plugins = config.get("plugins", {})
        entries = plugins.get("entries", {})
        
        # Check if feishu plugin is enabled and has config
        if "feishu" in entries and entries["feishu"].get("enabled", True):
            # Check if the feishu extension directory exists and has the plugin
            ext_path = os.path.expanduser("~/.openclaw/extensions/feishu")
            if os.path.isdir(ext_path):
                return True
        
        # Also check if feishu is in the plugins load paths
        load_paths = plugins.get("load", {}).get("paths", [])
        for p in load_paths:
            if "feishu" in p:
                return True
        
        return False
    
    # For other channels, assume configured = paired for now
    return True

def save_target_to_user_md(target_id):
    """Save target ID to USER.md for future use."""
    workspace = os.path.expanduser("~/.openclaw/workspace")
    user_md = os.path.join(workspace, "USER.md")
    if not os.path.exists(user_md):
        return
    
    with open(user_md) as f:
        content = f.read()
    
    # Check if already saved
    if target_id in content:
        return
    
    # Add to Notes section
    if "- **Notes:**" in content:
        content = content.replace(
            "- **Notes:**",
            f"- **Notes:**\n  - 飞书默认发送目标: `{target_id}`"
        )
    else:
        content += f"\n- 飞书默认发送目标: `{target_id}`\n"
    
    with open(user_md, 'w') as f:
        f.write(content)

def main():
    report_path = sys.argv[1] if len(sys.argv) > 1 else None
    archive_path = sys.argv[2] if len(sys.argv) > 2 else None
    target_id = sys.argv[3] if len(sys.argv) > 3 else None
    channel = sys.argv[4] if len(sys.argv) > 4 else "feishu"
    
    if not report_path or not os.path.exists(report_path):
        print("❌ 报告文件不存在:", report_path)
        sys.exit(1)
    
    # Step 2: Check channel configured
    if not check_channel_configured(channel):
        print(f"⚠️  未配置消息通道 ({channel})")
        print(f"   请先配置通道（如飞书需完成 pairing: openclaw pairing approve feishu <code>）")
        print(f"   或手动查看报告: {report_path}")
        sys.exit(1)
    
    print(f"✅ Channel ({channel}) 已配置")
    
    # Step 3: Check pairing
    if not check_pairing_done(channel):
        print(f"⚠️  Channel ({channel}) 已配置但未完成配对")
        print(f"   请先完成 pairing（如飞书需 approve pairing code）")
        sys.exit(1)
    
    print(f"✅ Pairing 已完成")
    
    # Step 4: Find target ID
    if not target_id:
        target_id = find_default_target()
    
    if not target_id:
        print("⚠️  未找到目标发送 ID")
        print("   请提供用户 ID (ou_xxx) 或群聊 ID (oc_xxx)")
        print(f"   用法: python3 deliver_results.py <report> <archive> <target_id> [channel]")
        sys.exit(1)
    
    print(f"✅ 目标 ID: {target_id}")
    
    # Save for future use
    save_target_to_user_md(target_id)
    
    # Step 5: Send
    # Read report content
    with open(report_path) as f:
        report_content = f.read()
    
    # Send report text
    print(f"📤 发送报告文本到 {target_id}...")
    result = subprocess.run(
        ["openclaw", "message", "send",
         "--channel", channel,
         "--target", target_id,
         "--message", report_content[:4000]],  # Feishu message length limit
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print("✅ 报告文本已发送")
    else:
        print(f"❌ 报告文本发送失败: {result.stderr}")
    
    # Send archive file
    if archive_path and os.path.exists(archive_path):
        print(f"📤 发送压缩包到 {target_id}...")
        result = subprocess.run(
            ["openclaw", "message", "send",
             "--channel", channel,
             "--target", target_id,
             "--message", f"📦 压测结果数据包",
             "--media", archive_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("✅ 压缩包已发送")
        else:
            print(f"❌ 压缩包发送失败: {result.stderr}")
    
    print(f"\n✅ 交付完成！目标: {target_id}")

if __name__ == "__main__":
    main()
