import asyncio


async def main():
    print("hello world")


print(type(main))
print(type(main()))

asyncio.run(main())
